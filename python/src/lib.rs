// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

//! Python bindings for zetasketch-rs
//!
//! This module provides Python bindings for the zetasketch-rs crate, which is a Rust
//! reimplementation of the ZetaSketch Java library for HyperLogLog++ implementation used
//! by Google BigQuery and BigTable.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use zetasketch_rs::{Aggregator, HyperLogLogPlusPlus as RustHyperLogLogPlusPlus, SketchError};

/// Convert Rust SketchError to Python exception
fn sketch_error_to_pyerr(err: SketchError) -> PyErr {
    match err {
        SketchError::IllegalArgument(msg) => {
            PyValueError::new_err(format!("Invalid argument: {}", msg))
        }
        _ => PyRuntimeError::new_err(format!("ZetaSketch error: {}", err)),
    }
}

/// Wrapper for the Rust HyperLogLogPlusPlus aggregator.
#[pyclass(name = "HyperLogLogPlusPlus")]
#[derive(Clone)]
pub struct PyHyperLogLogPlusPlus {
    inner: RustHyperLogLogPlusPlus,
}

#[pymethods]
impl PyHyperLogLogPlusPlus {
    #[new]
    #[pyo3(signature = (
            ty,
            normal_precision=RustHyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION,
            sparse_precision=Some(RustHyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION+RustHyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA)))
    ]
    pub fn constructor(
        ty: &Bound<'_, PyType>,
        normal_precision: i32,
        sparse_precision: Option<i32>,
    ) -> PyResult<Self> {
        let mut builder = RustHyperLogLogPlusPlus::builder().normal_precision(normal_precision);

        builder = match sparse_precision {
            Some(precision) => builder.sparse_precision(precision),
            None => builder.no_sparse_mode(),
        };

        let hll = match ty.name()?.to_str()? {
            "str" => builder.build_for_string(),
            "int" => builder.build_for_u64(),
            "bytes" => builder.build_for_bytes(),
            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported type for HyperLogLogPlusPlus. Use 'str', 'int', or 'bytes'.",
                ));
            }
        }
        .map_err(sketch_error_to_pyerr)?;

        Ok(Self { inner: hll })
    }

    /// Wrapper for the Rust [`HyperLogLogPlusPlus::from_bytes`] method.
    #[classmethod]
    pub fn from_bytes(_cls: &Bound<'_, PyType>, data: &[u8]) -> PyResult<Self> {
        let hll = RustHyperLogLogPlusPlus::from_bytes(data).map_err(sketch_error_to_pyerr)?;
        Ok(Self { inner: hll })
    }

    /// Wrapper for various `HyperLogLogPlusPlus::add_*` overloads.
    pub fn add(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(value) = value.extract::<i64>() {
            self.inner.add_i64(value)
        } else if let Ok(value) = value.extract::<&str>() {
            self.inner.add_string(value)
        } else if let Ok(value) = value.extract::<Vec<u8>>() {
            self.inner.add_bytes(&value)
        } else {
            return Err(PyTypeError::new_err(
                "Unsupported type for HyperLogLogPlusPlus",
            ));
        };

        match result {
            Ok(()) => Ok(()),
            // Explicitly map InvalidState to TypeError for better Python semantics
            Err(SketchError::InvalidState(msg)) => Err(PyTypeError::new_err(msg)),
            Err(err) => Err(sketch_error_to_pyerr(err)),
        }
    }

    /// Wrapper for the Rust [`HyperLogLogPlusPlus::merge_bytes`] method.
    pub fn merge(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(data) = data.extract::<Vec<u8>>() {
            self.inner.merge_bytes(&data).map_err(sketch_error_to_pyerr)
        } else if let Ok(other) = data.extract::<PyHyperLogLogPlusPlus>() {
            self.inner
                .merge_aggregator(other.inner)
                .map_err(sketch_error_to_pyerr)
        } else {
            Err(PyTypeError::new_err(
                "Unsupported type for HyperLogLogPlusPlus::merge",
            ))
        }
    }

    /// Get the estimated cardinality.
    ///
    /// Returns:
    ///     Estimated number of unique elements
    pub fn result(&self) -> PyResult<i64> {
        self.inner.result().map_err(sketch_error_to_pyerr)
    }

    /// Get the total number of values added to this sketch.
    ///
    /// Returns:
    ///     Number of values added
    pub fn num_values(&self) -> u64 {
        self.inner.num_values()
    }

    /// Serialize the sketch to bytes.
    ///
    /// Returns:
    ///     Serialized sketch data
    pub fn to_bytes(&self) -> PyResult<Vec<u8>> {
        self.inner
            .clone()
            .serialize_to_bytes()
            .map_err(sketch_error_to_pyerr)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "HyperLogLogPlusPlus(cardinality={}, num_values={})",
            self.result()?,
            self.num_values()
        ))
    }

    #[getter]
    pub fn normal_precision(&self) -> i32 {
        self.inner.normal_precision()
    }

    #[getter]
    pub fn sparse_precision(&self) -> i32 {
        self.inner.sparse_precision()
    }
}

/// Python module for ZetaSketch HyperLogLog++ implementation
#[pymodule]
fn zetasketch_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHyperLogLogPlusPlus>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
