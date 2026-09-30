use ::ckmeans::CkmeansConfig;
use ::ckmeans::CkmeansErr;
use ::ckmeans::ckmeans_indices;
use ::ckmeans::ckmeans_optimal;
use ::ckmeans::roundbreaks as rndb;
use numpy::AllowTypeChange;
use numpy::PyArray1;
use numpy::PyArrayLike1;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PySlice;
use pyo3::wrap_pyfunction;

create_exception!(
    ckmeans,
    CkmeansError,
    PyValueError,
    "Raised when the input data or the number of clusters is not valid."
);

/// Convert a crate error to a Python exception.
///
/// Errors that the caller can cause become `CkmeansError`. Errors that come
/// from the internal calculation become `RuntimeError`.
fn to_pyerr(err: CkmeansErr) -> PyErr {
    match err {
        CkmeansErr::TooFewClassesError
        | CkmeansErr::TooManyClassesError
        | CkmeansErr::InvalidRangeError
        | CkmeansErr::NanError => CkmeansError::new_err(err.to_string()),
        CkmeansErr::ConversionError
        | CkmeansErr::LowWindowError
        | CkmeansErr::HighWindowError
        | CkmeansErr::InfallibleError => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Convert a cluster count to the `u8` that the crate uses.
fn to_nclusters(k: i64, name: &str) -> PyResult<u8> {
    u8::try_from(k)
        .map_err(|_| CkmeansError::new_err(format!("{name} must be from 1 to 255, not {k}")))
}

/// Input data: any object that NumPy can convert to a 1D `float64` array.
type Data<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;

/// Copy the input data into a `Vec`.
///
/// The copy lets the calculation run without the GIL, because other threads
/// cannot then change the data during the calculation. It also accepts arrays
/// that are not contiguous in memory.
fn to_values(data: &Data<'_>) -> Vec<f64> {
    data.as_array().to_vec()
}

/// Cluster data into k groups with the least within-group sum of squares.
///
/// The algorithm is the dynamic programme of Wang & Song (2011). The groups are
/// optimally homogeneous, which makes them useful to show a continuous
/// variable as discrete colour or style classes.
///
/// Parameters
/// ----------
/// data : array_like
///     One-dimensional data. The data is converted to float64.
/// k : int
///     The number of clusters, from 1 to 255.
///
/// Returns
/// -------
/// list of numpy.ndarray
///     The clusters, in ascending order of value. Each cluster is sorted, and
///     is a view of one sorted copy of the data. If the data has fewer than k
///     distinct values, there is one cluster for each distinct value.
///
/// Raises
/// ------
/// CkmeansError
///     If k is less than 1, greater than 255 or greater than the number of
///     values, or if the data contains NaN.
#[pyfunction]
#[pyo3(name = "ckmeans", signature = (data, /, k))]
fn ckmeans_wrapper<'a>(
    py: Python<'a>,
    data: Data<'a>,
    k: i64,
) -> PyResult<Vec<Bound<'a, PyArray1<f64>>>> {
    let nclusters = to_nclusters(k, "k")?;
    let values = to_values(&data);
    let (sorted, ranges) = py
        .detach(|| ckmeans_indices(&values, nclusters))
        .map_err(to_pyerr)?;
    // Each cluster is a view of one sorted array. The indices are less than
    // the length of a slice, so they are not greater than isize::MAX.
    let sorted = PyArray1::from_vec(py, sorted);
    ranges
        .into_iter()
        .map(|(start, end)| {
            let slice = PySlice::new(py, start as isize, end as isize + 1, 1);
            Ok(sorted.get_item(slice)?.cast_into::<PyArray1<f64>>()?)
        })
        .collect()
}

/// Calculate the breaks between k clusters, for labels and legends.
///
/// Each break b is between the highest value of a cluster (last) and the lowest
/// value of the next cluster (first): last < b <= first. The break is the
/// roundest number in that interval, so a legend shows only the precision
/// that is necessary to separate the clusters. The method is based on the
/// visionscarto natural-breaks method.
///
/// Parameters
/// ----------
/// data : array_like
///     One-dimensional data. The data is converted to float64.
/// k : int
///     The number of clusters, from 1 to 255.
///
/// Returns
/// -------
/// numpy.ndarray
///     One break fewer than the number of clusters. If the data has fewer than
///     k distinct values, there are fewer than k - 1 breaks.
///
/// Raises
/// ------
/// CkmeansError
///     If k is less than 1, greater than 255 or greater than the number of
///     values, or if the data contains NaN.
#[pyfunction]
#[pyo3(name = "breaks", signature = (data, /, k))]
fn roundbreaks_wrapper<'a>(
    py: Python<'a>,
    data: Data<'a>,
    k: i64,
) -> PyResult<Bound<'a, PyArray1<f64>>> {
    let nclusters = to_nclusters(k, "k")?;
    let values = to_values(&data);
    let result = py.detach(|| rndb(&values, nclusters)).map_err(to_pyerr)?;
    Ok(PyArray1::from_vec(py, result))
}

/// The result of `ckmeans_optimal`.
#[pyclass(frozen, get_all, module = "ckmeans")]
struct OptimalResult {
    /// The chosen number of clusters.
    k: u8,
    /// The clusters, in ascending order of value.
    clusters: Py<PyList>,
    /// The mean of each cluster.
    centers: Py<PyArray1<f64>>,
    /// The number of values in each cluster.
    sizes: Py<PyArray1<isize>>,
    /// The within-cluster sum of squares of each cluster.
    withinss: Py<PyArray1<f64>>,
    /// The candidate numbers of clusters.
    ks: Py<PyArray1<isize>>,
    /// The BIC of each candidate number of clusters.
    bic: Py<PyArray1<f64>>,
}

#[pymethods]
impl OptimalResult {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "OptimalResult(k={}, centers={}, sizes={}, withinss={}, ks={}, bic={})",
            self.k,
            self.centers.bind(py).repr()?,
            self.sizes.bind(py).repr()?,
            self.withinss.bind(py).repr()?,
            self.ks.bind(py).repr()?,
            self.bic.bind(py).repr()?,
        ))
    }
}

/// Cluster data with the number of clusters that has the lowest BIC.
///
/// The function clusters the data for each k from k_min to k_max and chooses
/// the k with the lowest Bayesian Information Criterion (Song & Zhong 2020).
/// k_max is capped at the number of distinct values in the data.
///
/// Parameters
/// ----------
/// data : array_like
///     One-dimensional data. The data is converted to float64.
/// k_min : int, default 1
///     The lowest number of clusters to evaluate, from 1 to 255.
/// k_max : int, default 9
///     The highest number of clusters to evaluate, from 1 to 255.
///
/// Returns
/// -------
/// OptimalResult
///     The chosen k, its clusters and their statistics, and the BIC of each
///     candidate k. If the data contains an infinite value, each BIC is NaN
///     and the result uses k_min.
///
/// Raises
/// ------
/// CkmeansError
///     If k_min is less than 1, k_min is greater than k_max, k_min is greater
///     than the number of distinct values, either value is greater than 255,
///     or the data contains NaN.
#[pyfunction]
#[pyo3(name = "ckmeans_optimal", signature = (data, /, k_min = 1, k_max = 9))]
fn ckmeans_optimal_wrapper(
    py: Python<'_>,
    data: Data<'_>,
    k_min: i64,
    k_max: i64,
) -> PyResult<OptimalResult> {
    let config = CkmeansConfig {
        k_min: to_nclusters(k_min, "k_min")?,
        k_max: to_nclusters(k_max, "k_max")?,
    };
    let values = to_values(&data);
    let result = py
        .detach(|| ckmeans_optimal(&values, config))
        .map_err(to_pyerr)?;
    let clusters = PyList::new(
        py,
        result
            .clusters
            .into_iter()
            .map(|cluster| PyArray1::from_vec(py, cluster)),
    )?;
    // A cluster size cannot be greater than isize::MAX, because it is the
    // length of a slice.
    let sizes: Vec<isize> = result.stats.iter().map(|s| s.size as isize).collect();
    let centers: Vec<f64> = result.stats.iter().map(|s| s.center).collect();
    let withinss: Vec<f64> = result.stats.iter().map(|s| s.withinss).collect();
    let (ks, bic): (Vec<isize>, Vec<f64>) = result
        .bic
        .iter()
        .map(|&(k, bic)| (isize::from(k), bic))
        .unzip();
    Ok(OptimalResult {
        k: result.k,
        clusters: clusters.unbind(),
        centers: PyArray1::from_vec(py, centers).unbind(),
        sizes: PyArray1::from_vec(py, sizes).unbind(),
        withinss: PyArray1::from_vec(py, withinss).unbind(),
        ks: PyArray1::from_vec(py, ks).unbind(),
        bic: PyArray1::from_vec(py, bic).unbind(),
    })
}

#[pymodule]
fn _ckmeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CkmeansError", m.py().get_type::<CkmeansError>())?;
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(ckmeans_optimal_wrapper, m)?)?;
    m.add_class::<OptimalResult>()?;
    Ok(())
}
