use ::ckmeans::CkmeansErr;
use ::ckmeans::ckmeans as ckm;
use ::ckmeans::roundbreaks as rndb;
use numpy::PyArray1;
use numpy::borrow::PyReadonlyArray1;
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

#[pyfunction]
#[pyo3(name = "ckmeans")]
#[pyo3(text_signature = "ckmeans(data, k, /)
--
Cluster data into k bins

Minimizing the difference within groups – what Wang & Song refer to as withinss,
or within sum-of-squares, means that groups are optimally homogenous within and the data are
split into representative groups. This is very useful for visualization, where one may wish to
represent a continuous variable in discrete colour or style groups. This function can provide
groups – or “classes” – that emphasize differences between data.")]
fn ckmeans_wrapper<'a>(
    py: Python<'a>,
    data: PyReadonlyArray1<'a, f64>,
    k: i64,
) -> PyResult<Vec<Bound<'a, PyArray1<f64>>>> {
    let nclusters = to_nclusters(k, "k")?;
    // A copy also accepts arrays that are not contiguous in memory.
    let values = data.as_array().to_vec();
    let result = ckm(&values, nclusters).map_err(to_pyerr)?;
    Ok(result
        .into_iter()
        .map(|v| PyArray1::from_vec(py, v))
        .collect())
}

#[pyfunction]
#[pyo3(name = "breaks")]
#[pyo3(text_signature = "breaks(data, k, /)
--
Calculate k - 1 breaks in the data, distinguishing classes for labelling or visualisation

The boundaries of the classes returned by ckmeans are “ugly” in the sense that the values
returned are the lower bound of each cluster, which can’t be used for labelling, since they
might have many decimal places. To create a legend, the values should be rounded — but the
rounding might be either too loose (and would result in spurious decimal places), or too
strict, resulting in classes ranging “from x to x”. A better approach is to choose the roundest
number that separates the lowest point from a class from the highest point in the preceding
class — thus giving just enough precision to distinguish the classes.
This function is closer to what Jenks returns: k - 1 “breaks” in the data, useful for labelling.")]
fn roundbreaks_wrapper<'a>(
    py: Python<'a>,
    data: PyReadonlyArray1<'a, f64>,
    k: i64,
) -> PyResult<Bound<'a, PyArray1<f64>>> {
    let nclusters = to_nclusters(k, "k")?;
    let values = data.as_array().to_vec();
    let result = rndb(&values, nclusters).map_err(to_pyerr)?;
    Ok(PyArray1::from_vec(py, result))
}

#[pymodule]
fn _ckmeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CkmeansError", m.py().get_type::<CkmeansError>())?;
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, m)?)?;
    Ok(())
}
