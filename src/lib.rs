use ::ckmeans::ckmeans as ckm;
use ::ckmeans::roundbreaks as rndb;
use numpy::PyArray1;
use numpy::borrow::PyReadonlyArray1;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, wrap_pyfunction};

// Define custom Python exceptions
create_exception!(ckmeans, CkmeansError, PyException);
create_exception!(ckmeans, InvalidDataError, CkmeansError);
create_exception!(ckmeans, InvalidClusterCountError, CkmeansError);
create_exception!(ckmeans, ComputationError, CkmeansError);

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
    // Get the array slice
    let array_view = data
        .as_slice()
        .map_err(|_| InvalidDataError::new_err("Failed to read input data array"))?;

    // Validate input data
    if array_view.is_empty() {
        return Err(InvalidDataError::new_err("Input data array is empty"));
    }

    // Validate k value
    if k == 0 {
        return Err(InvalidClusterCountError::new_err(
            "Number of clusters (k) must be greater than 0",
        ));
    }

    if k > array_view.len() {
        return Err(InvalidClusterCountError::new_err(format!(
            "Number of clusters ({}) cannot exceed the number of data points ({})",
            k,
            array_view.len()
        )));
    }

    // Check for NaN or infinite values
    for (i, &value) in array_view.iter().enumerate() {
        if value.is_nan() {
            return Err(InvalidDataError::new_err(format!(
                "Input data contains NaN at index {i}"
            )));
        }
        if value.is_infinite() {
            return Err(InvalidDataError::new_err(format!(
                "Input data contains infinite value at index {i}"
            )));
        }
    }

    // Convert k to u8 safely
    let k_u8 = k
        .try_into()
        .map_err(|_| InvalidClusterCountError::new_err("Number of clusters is too large"))?;

    // Call the ckmeans function
    match ckm(array_view, k_u8) {
        Ok(result) => {
            let flattened: Vec<_> = result
                .into_iter()
                .map(|v| PyArray1::from_vec(py, v).to_owned())
                .collect();
            Ok(flattened)
        }
        Err(err) => Err(ComputationError::new_err(format!(
            "Failed to compute clusters: {err}"
        ))),
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
    // Get the array slice
    let array_view = data
        .as_slice()
        .map_err(|_| InvalidDataError::new_err("Failed to read input data array"))?;

    // Validate input data
    if array_view.is_empty() {
        return Err(InvalidDataError::new_err("Input data array is empty"));
    }

    // Validate k value
    if k == 0 {
        return Err(InvalidClusterCountError::new_err(
            "Number of clusters (k) must be greater than 0",
        ));
    }

    if k > array_view.len() {
        return Err(InvalidClusterCountError::new_err(format!(
            "Number of clusters ({}) cannot exceed the number of data points ({})",
            k,
            array_view.len()
        )));
    }

    // Check for NaN or infinite values
    for (i, &value) in array_view.iter().enumerate() {
        if value.is_nan() {
            return Err(InvalidDataError::new_err(format!(
                "Input data contains NaN at index {i}"
            )));
        }
        if value.is_infinite() {
            return Err(InvalidDataError::new_err(format!(
                "Input data contains infinite value at index {i}"
            )));
        }
    }

    // Convert k to u8 safely
    let k_u8 = k
        .try_into()
        .map_err(|_| InvalidClusterCountError::new_err("Number of clusters is too large"))?;

    // Call the roundbreaks function
    match rndb(array_view, k_u8) {
        Ok(result) => Ok(PyArray1::from_vec(py, result).into()),
        Err(err) => Err(ComputationError::new_err(format!(
            "Failed to compute breaks: {err}"
        ))),
    }
}

#[pymodule]
fn ckmeans(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ckmeans_wrapper, &m)?)?;
    m.add_function(wrap_pyfunction!(roundbreaks_wrapper, &m)?)?;
    m.add("CkmeansError", py.get_type::<CkmeansError>())?;
    m.add("InvalidDataError", py.get_type::<InvalidDataError>())?;
    m.add(
        "InvalidClusterCountError",
        py.get_type::<InvalidClusterCountError>(),
    )?;
    m.add("ComputationError", py.get_type::<ComputationError>())?;
    Ok(())
}
