use ::ckmeans::ckmeans as ckm;
use ::ckmeans::roundbreaks as rndb;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[pyo3(name = "ckmeans")]
/// ckmeans(data, k, /)
/// --
///
/// Cluster data into k number of bins
///
/// Minimizing the difference within groups – what Wang & Song refer to as withinss,
/// or within sum-of-squares, means that groups are optimally homogenous within and the data are
/// split into representative groups. This is very useful for visualization, where one may wish to
/// represent a continuous variable in discrete colour or style groups. This function can provide
/// groups – or “classes” – that emphasize differences between data.
fn ckmeans_wrapper(data: Vec<f64>, k: usize) -> PyResult<Vec<Vec<f64>>> {
    match ckm(&data, k.try_into().unwrap()) {
        Ok(result) => Ok(result),
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            err
        ))),
    }
}

#[pyfunction]
#[pyo3(name = "breaks")]
/// breaks(data, k, /)
/// --
///
/// Calculate k - 1 breaks in the data, distinguishing classes for labelling or visualisation
///
/// The boundaries of the classes returned by ckmeans are “ugly” in the sense that the values
/// returned are the lower bound of each cluster, which can’t be used for labelling, since they
/// might have many decimal places. To create a legend, the values should be rounded — but the
/// rounding might be either too loose (and would result in spurious decimal places), or too
/// strict, resulting in classes ranging “from x to x”. A better approach is to choose the roundest
/// number that separates the lowest point from a class from the highest point in the preceding
/// class — thus giving just enough precision to distinguish the classes.
/// This function is closer to what Jenks returns: k - 1 “breaks” in the data, useful for labelling.
fn roundbreaks_wrapper(data: Vec<f64>, k: usize) -> PyResult<Vec<f64>> {
    match rndb(&data, k.try_into().unwrap()) {
        Ok(result) => Ok(result),
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "{}",
            err
        ))),
    }
}

#[pymodule]
fn ckmeans(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, m)?)?;
    Ok(())
}
