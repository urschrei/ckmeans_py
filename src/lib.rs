use ::ckmeans::ckmeans as ckm;
use ::ckmeans::roundbreaks as rndb;
use numpy::borrow::PyReadonlyArray1;
use numpy::PyArray1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

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
    k: usize,
) -> PyResult<Vec<Bound<'a, PyArray1<f64>>>> {
    let array_view = data.as_slice().unwrap();
    match ckm(array_view, k.try_into().unwrap()) {
        Ok(result) => {
            let flattened: Vec<_> = result
                .into_iter()
                .map(|v| PyArray1::from_vec(py, v).to_owned())
                .collect();
            Ok(flattened)
        }
        Err(err) => Err(PyRuntimeError::new_err(format!("{err}"))),
    }
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
fn roundbreaks_wrapper(
    py: Python,
    data: PyReadonlyArray1<f64>,
    k: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let array_view = data.as_slice().unwrap();
    match rndb(array_view, k.try_into().unwrap()) {
        Ok(result) => Ok(PyArray1::from_vec(py, result).into()),
        Err(err) => Err(PyRuntimeError::new_err(format!("{err}"))),
    }
}

#[pymodule]
fn ckmeans(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, &m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, &m)?)?;
    Ok(())
}
