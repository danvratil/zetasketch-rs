// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

// Replicates com.google.zetasketch.internal.hllplus.Representation.java (the abstract class)

use crate::error::SketchError;
use crate::hll::normal_representation::NormalRepresentation;
use crate::hll::sparse_representation::SparseRepresentation;
use crate::hll::state::State;

/// Value used to indicate that the sparse representation should not be used.
pub const SPARSE_PRECISION_DISABLED: i32 = 0;

/// Trait defining common operations for HLL representations.
pub trait RepresentationOps: std::fmt::Debug {
    /// Adds a hashed value to the representation.
    /// Returns `Ok(Some(new_representation))` if the representation type changes (e.g., sparse to normal).
    /// Returns `Ok(None)` if the representation type remains the same.
    fn add_hash(self, hash: u64) -> Result<RepresentationUnion, SketchError>;

    /// Adds a single sparse-encoded value.
    fn add_sparse_value(
        self,
        encoding: &crate::hll::encoding::Sparse,
        sparse_value: u32,
    ) -> Result<RepresentationUnion, SketchError>;

    /// Adds multiple sparse-encoded values from an iterator.
    fn add_sparse_values<I: IntoIterator<Item = u32>>(
        self,
        encoding: &crate::hll::encoding::Sparse,
        sparse_values: Option<I>,
    ) -> Result<RepresentationUnion, SketchError>;

    /// Estimates the cardinality from the current representation.
    fn estimate(&self) -> Result<i64, SketchError>;

    /// Merges another NormalRepresentation into this one.
    fn merge_from_normal(
        self,
        other: NormalRepresentation,
    ) -> Result<RepresentationUnion, SketchError>;

    /// Merges another SparseRepresentation into this one.
    fn merge_from_sparse(
        self,
        other: SparseRepresentation,
    ) -> Result<RepresentationUnion, SketchError>;

    /// Makes the representation as compact as possible.
    /// May change representation type (e.g. sparse buffer flush leading to normal).
    fn compact(self) -> Result<RepresentationUnion, SketchError>;

    /// Provides mutable access to the underlying state.
    fn state_mut(&mut self) -> &mut State;

    /// Provides immutable access to the underlying state.
    fn state(&self) -> &State;
}

/// Enum to dispatch calls to the appropriate representation.
#[derive(Debug, Clone)]
pub enum RepresentationUnion {
    Normal(NormalRepresentation),
    Sparse(SparseRepresentation),
    Invalid,
}

impl RepresentationUnion {
    pub fn state(&self) -> &State {
        match self {
            RepresentationUnion::Normal(n) => n.state(),
            RepresentationUnion::Sparse(s) => s.state(),
            RepresentationUnion::Invalid => {
                panic!("Invalid representation");
            }
        }
    }

    pub fn state_mut(&mut self) -> &mut State {
        match self {
            RepresentationUnion::Normal(n) => n.state_mut(),
            RepresentationUnion::Sparse(s) => s.state_mut(),
            RepresentationUnion::Invalid => panic!("Invalid representation"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Representation {
    repr: RepresentationUnion,
}

impl Representation {
    pub fn from_state(state: State) -> Result<Self, SketchError> {
        if state.sparse_data.is_some() && state.sparse_precision == SPARSE_PRECISION_DISABLED {
            return Err(SketchError::InvalidState(
                "Must have a sparse precision when sparse data is set".to_string(),
            ));
        }

        if state.data.is_some() || state.sparse_precision == SPARSE_PRECISION_DISABLED {
            Ok(Self {
                repr: RepresentationUnion::Normal(NormalRepresentation::new(state)?),
            })
        } else {
            Ok(Self {
                repr: RepresentationUnion::Sparse(SparseRepresentation::new(state)?),
            })
        }
    }

    pub fn add_hash(&mut self, hash: u64) -> Result<(), SketchError> {
        self.repr = match std::mem::replace(&mut self.repr, RepresentationUnion::Invalid) {
            RepresentationUnion::Normal(n) => n.add_hash(hash)?,
            RepresentationUnion::Sparse(s) => s.add_hash(hash)?,
            RepresentationUnion::Invalid => {
                return Err(SketchError::InvalidState(
                    "Representation is invalid".to_string(),
                ))
            }
        };
        Ok(())
    }

    pub fn estimate(&self) -> Result<i64, SketchError> {
        match &self.repr {
            RepresentationUnion::Normal(n) => n.estimate(),
            RepresentationUnion::Sparse(s) => s.estimate(),
            RepresentationUnion::Invalid => Err(SketchError::InvalidState(
                "Representation is invalid".to_string(),
            )),
        }
    }

    pub fn compact(&mut self) -> Result<(), SketchError> {
        self.repr = match std::mem::replace(&mut self.repr, RepresentationUnion::Invalid) {
            RepresentationUnion::Normal(n) => n.compact()?,
            RepresentationUnion::Sparse(s) => s.compact()?,
            RepresentationUnion::Invalid => {
                return Err(SketchError::InvalidState(
                    "Representation is invalid".to_string(),
                ))
            }
        };
        Ok(())
    }

    pub fn state(&self) -> &State {
        self.repr.state()
    }

    pub fn state_mut(&mut self) -> &mut State {
        self.repr.state_mut()
    }

    pub fn merge(&mut self, other: Representation) -> Result<(), SketchError> {
        // FIXME: This is sub-optimal, but it is the only way to prevent the current sketch
        // from being corrupted if the merge fails mid-way.
        self.repr = match (self.repr.clone(), other.repr) {
            (RepresentationUnion::Normal(n1), RepresentationUnion::Normal(n2)) => {
                n1.merge_from_normal(n2)?
            }
            (RepresentationUnion::Sparse(s1), RepresentationUnion::Sparse(s2)) => {
                s1.merge_from_sparse(s2)?
            }
            (RepresentationUnion::Normal(n), RepresentationUnion::Sparse(s)) => {
                n.merge_from_sparse(s)?
            }
            (RepresentationUnion::Sparse(s), RepresentationUnion::Normal(n)) => {
                s.merge_from_normal(n)?
            }
            _ => {
                return Err(SketchError::InvalidState(
                    "Representation is invalid".to_string(),
                ))
            }
        };
        Ok(())
    }
}

#[cfg(test)]
impl Representation {
    pub fn is_normal(&self) -> bool {
        matches!(self.repr, RepresentationUnion::Normal(_))
    }

    pub fn is_sparse(&self) -> bool {
        matches!(self.repr, RepresentationUnion::Sparse(_))
    }
}
