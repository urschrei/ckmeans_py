use ::ckmeans::CkmeansErr;
use ::ckmeans::ckmeans as ckm;
use ::ckmeans::roundbreaks as rndb;
use numpy::AllowTypeChange;
use numpy::PyArray1;
use numpy::PyArrayLike1;
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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
///     The clusters, in ascending order of value. Each cluster is sorted. If
///     the data has fewer than k distinct values, there is one cluster for each
///     distinct value.
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
    let result = py.detach(|| ckm(&values, nclusters)).map_err(to_pyerr)?;
    Ok(result
        .into_iter()
        .map(|v| PyArray1::from_vec(py, v))
        .collect())
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

#[pymodule]
fn _ckmeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CkmeansError", m.py().get_type::<CkmeansError>())?;
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, m)?)?;
    Ok(())
}
