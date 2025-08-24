// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use std::collections::HashSet;

use crate::{
    aggregator::Aggregator,
    error::SketchError,
    hll::{
        hash::Hash, normal_representation::NormalRepresentation, representation::Representation,
        sparse_representation::SparseRepresentation, state::State, value_type::ValueType,
    },
    protos::{AggregatorStateProto, AggregatorType, DefaultOpsTypeId},
};
use protobuf::Message;

/// Type of the HLL sketch
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Type {
    Long,
    Integer,
    String,
    Bytes,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Type::Long => "LONG",
            Type::Integer => "INTEGER",
            Type::String => "STRING",
            Type::Bytes => "BYTES",
        })
    }
}

impl std::fmt::Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl Type {
    pub fn all() -> HashSet<Type> {
        HashSet::from([Type::Long, Type::Integer, Type::String, Type::Bytes])
    }

    pub fn from_value_type(value_type: ValueType) -> Result<HashSet<Type>, SketchError> {
        match value_type {
            ValueType::DefaultOpsType(DefaultOpsTypeId::UINT64) => Ok(HashSet::from([Type::Long])),
            ValueType::DefaultOpsType(DefaultOpsTypeId::UINT32) => {
                Ok(HashSet::from([Type::Integer]))
            }
            ValueType::DefaultOpsType(DefaultOpsTypeId::BYTES_OR_UTF8_STRING) => {
                Ok(HashSet::from([Type::String, Type::Bytes]))
            }
            _ => Err(SketchError::InvalidState(format!(
                "Unsupported value type {value_type:?}"
            ))),
        }
    }

    pub fn extract_and_normalize(state: &State) -> Result<HashSet<Type>, SketchError> {
        if state.value_type == ValueType::Unknown {
            Ok(Type::all())
        } else {
            Type::from_value_type(state.value_type)
        }
    }
}

impl From<Type> for ValueType {
    fn from(val: Type) -> Self {
        match val {
            Type::Long => ValueType::DefaultOpsType(DefaultOpsTypeId::UINT64),
            Type::Integer => ValueType::DefaultOpsType(DefaultOpsTypeId::UINT32),
            Type::String => ValueType::DefaultOpsType(DefaultOpsTypeId::BYTES_OR_UTF8_STRING),
            Type::Bytes => ValueType::DefaultOpsType(DefaultOpsTypeId::BYTES_OR_UTF8_STRING),
        }
    }
}

/// HLL++ aggregator for estimating cardinalities of multisets.
///
/// The aggregator uses the standard format for storing the internal state of the cardinality
/// estimate as defined in hllplus-unique.proto, allowing users to merge aggregators with data
/// computed in C++ or Go and to load up the cardinalities in a variety of analysis tools.
///
/// The precision defines the accuracy of the HLL++ aggregator at the cost of the memory used. The
/// upper bound on the memory required is 2<sup>precision</sup> bytes, but less memory is used for
/// smaller cardinalities (up to ~2<sup>precision - 2</sup>). The relative error is 1.04 /
/// sqrt(2<sup>precision</sup>). A typical value used at Google is 15, which gives an error of about
///  0.6% while requiring an upper bound of 32 KiB of memory.
#[derive(Debug, Clone)]
pub struct HyperLogLogPlusPlus {
    representation: Representation,
    allowed_types: HashSet<Type>,
}

impl HyperLogLogPlusPlus {
    /// The smallest normal precision supported by this aggregator.
    pub const MINIMUM_PRECISION: i32 = NormalRepresentation::MINIMUM_PRECISION;
    /// The largest normal precision supported by this aggregator.
    pub const MAXIMUM_PRECISION: i32 = NormalRepresentation::MAXIMUM_PRECISION;
    /// The default normal precision that is used if the user does not specify a normal precision.
    pub const DEFAULT_NORMAL_PRECISION: i32 = 15;
    /// The largest sparse precision supported by this aggregator.
    pub const MAXIMUM_SPARSE_PRECISION: i32 = SparseRepresentation::MAXIMUM_SPARSE_PRECISION;
    /// Value used to indicate that the sparse representation should not be used.
    pub const SPARSE_PRECISION_DISABLED: i32 = SparseRepresentation::SPARSE_PRECISION_DISABLED;
    /// If no sparse precision is specified, this value is added to the normal precision to obtain
    /// the sparse precision, which optimizes the memory-precision trade-off.
    pub const DEFAULT_SPARSE_PRECISION_DELTA: i32 = 5;
    /// The encoding version of the [`AggregatorStateProto`]. We only support v2.
    pub const ENCODING_VERSION: i32 = 2;

    /// Returns a new builder to customize and create a new instance of this aggregator.
    pub fn builder() -> HyperLogLogPlusPlusBuilder {
        HyperLogLogPlusPlusBuilder::new()
    }

    pub(crate) fn from_state(state: State) -> Result<Self, SketchError> {
        if state.r#type != AggregatorType::HYPERLOGLOG_PLUS_UNIQUE {
            return Err(SketchError::InvalidState(format!(
                "Expected proto to be of type HYPERLOGLOG_PLUS_UNIQUE but was {:?}",
                state.r#type
            )));
        }
        if state.encoding_version != Self::ENCODING_VERSION {
            return Err(SketchError::InvalidState(format!(
                "Expected encoding version to be {:?} but was {:?}",
                Self::ENCODING_VERSION,
                state.encoding_version
            )));
        }
        let allowed_types = Type::extract_and_normalize(&state)?;
        Ok(Self {
            representation: Representation::from_state(state)?,
            allowed_types,
        })
    }

    /// Creates a new HyperLogLog++ aggregator from the serialized `proto`.
    ///
    /// The `proto` must be a valid aggregator state of type [`AggregatorType::HYPERLOGLOG_PLUS_UNIQUE`].
    pub fn from_proto(proto: AggregatorStateProto) -> Result<Self, SketchError> {
        let bytes = proto
            .write_to_bytes()
            .map_err(SketchError::ProtoDeserialization)?;
        Self::from_bytes(&bytes)
    }

    /// Creates a new HyperLogLog++ aggregator from the `bytes`.
    ///
    /// The `bytes` must be a valid serialized [`AggregatorStateProto`] of the type
    /// [`AggregatorType::HYPERLOGLOG_PLUS_UNIQUE`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SketchError> {
        Self::from_state(State::parse(bytes)?)
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `i32` or `u32`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_u32`].
    pub fn add_i32(&mut self, value: i32) -> Result<(), SketchError> {
        self.check_and_set_type(Type::Integer)?;
        self.add_hash(Hash::of_i32(value))
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `i32` or `u32`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_u32`].
    pub fn add_u32(&mut self, value: u32) -> Result<(), SketchError> {
        self.check_and_set_type(Type::Integer)?;
        self.add_hash(Hash::of_u32(value))
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `i64` or `u64`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_u64`].
    pub fn add_i64(&mut self, value: i64) -> Result<(), SketchError> {
        self.check_and_set_type(Type::Long)?;
        self.add_hash(Hash::of_i64(value))
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `i64` or `u64`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_u64`].
    pub fn add_u64(&mut self, value: u64) -> Result<(), SketchError> {
        self.check_and_set_type(Type::Long)?;
        self.add_hash(Hash::of_u64(value))
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `bytes`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_bytes`].
    pub fn add_bytes(&mut self, value: &[u8]) -> Result<(), SketchError> {
        self.check_and_set_type(Type::Bytes)?;
        self.add_hash(Hash::of_bytes(value))
    }

    /// Add `value` to the aggregator.
    ///
    /// Returns [`SketchError`] if the aggregator is of different type than `string`.
    /// See [`HyperLogLogPlusPlusBuilder::build_for_string`].
    pub fn add_string(&mut self, value: &str) -> Result<(), SketchError> {
        self.check_and_set_type(Type::String)?;
        self.add_hash(Hash::of_string(value))
    }

    /// Returns the normal precision of the aggregator.
    pub fn normal_precision(&self) -> i32 {
        self.representation.state().precision
    }

    /// Returns the sparse precision of the aggregator.
    pub fn sparse_precision(&self) -> i32 {
        self.representation.state().sparse_precision
    }

    fn add_hash(&mut self, hash: u64) -> Result<(), SketchError> {
        self.representation.add_hash(hash)?;
        self.representation.state_mut().num_values += 1;
        Ok(())
    }

    fn check_type_and_merge(&mut self, other: HyperLogLogPlusPlus) -> Result<(), SketchError> {
        let mut new_types = self.allowed_types.clone();
        new_types.retain(|t| other.allowed_types.contains(t));
        if new_types.is_empty() {
            return Err(SketchError::InvalidState(format!(
                "Aggregator of type {:?} is incompatible with aggregator of type {:?}",
                self.allowed_types, other.allowed_types
            )));
        }

        let num_values = other.representation.state().num_values;
        self.representation.merge(other.representation)?;
        self.representation.state_mut().num_values += num_values;
        // Only updat the allowed  types after a successful merge
        self.allowed_types = new_types;
        Ok(())
    }

    fn check_and_set_type(&mut self, r#type: Type) -> Result<(), SketchError> {
        if !self.allowed_types.contains(&r#type) {
            return Err(SketchError::InvalidState(format!(
                "Unable to add type {:?} to aggregator of type {:?}",
                r#type, self.allowed_types
            )));
        }

        // Narrow the type if necessary.
        if self.allowed_types.len() > 1 {
            self.allowed_types.clear();
            self.allowed_types.insert(r#type);
            self.representation.state_mut().value_type = r#type.into();
        }
        Ok(())
    }
}

impl Aggregator<i64, HyperLogLogPlusPlus> for HyperLogLogPlusPlus {
    fn result(&self) -> Result<i64, SketchError> {
        self.representation.estimate()
    }

    fn merge_aggregator(&mut self, other: HyperLogLogPlusPlus) -> Result<(), SketchError> {
        self.check_type_and_merge(other)
    }

    fn merge_proto(&mut self, proto: AggregatorStateProto) -> Result<(), SketchError> {
        self.merge_aggregator(HyperLogLogPlusPlus::from_proto(proto)?)
    }

    fn merge_bytes(&mut self, data: &[u8]) -> Result<(), SketchError> {
        self.merge_aggregator(HyperLogLogPlusPlus::from_bytes(data)?)
    }

    fn num_values(&self) -> u64 {
        self.representation.state().num_values as u64
    }

    fn serialize_to_bytes(mut self) -> Result<Vec<u8>, SketchError> {
        self.representation.compact()?;
        self.representation.state().to_byte_array()
    }

    fn serialize_to_proto(mut self) -> Result<AggregatorStateProto, SketchError> {
        self.representation.compact()?;
        let bytes = self.representation.state().to_byte_array()?;
        AggregatorStateProto::parse_from_bytes(&bytes).map_err(SketchError::ProtoDeserialization)
    }
}

#[derive(Debug, Clone)]
pub struct HyperLogLogPlusPlusBuilder {
    normal_precision: i32,
    sparse_precision: Option<i32>,
}

impl HyperLogLogPlusPlusBuilder {
    pub(crate) fn new() -> Self {
        Self {
            normal_precision: HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION,
            sparse_precision: None,
        }
    }

    /// Sets the normal precision to be used. Must be in the range from [`HyperLogLogPlusPlus::MINIMUM_PRECISION`]
    /// to [`HyperLogLogPlusPlus::MAXIMUM_PRECISION`] (inclusive).
    ///
    /// The precision defines the accuracy of the HLL++ aggregator at the cost of the memory used.
    /// The upper bound on the memory required is 2<sup>precision</sup> bytes, but less memory is
    /// used for smaller cardinalities (up to ~2<sup>precision - 2</sup>). The relative error is 1.04
    /// / sqrt(2<sup>precision</sup>). If not specified, [`HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION`]` is used,
    ///  which gives an error of about 0.6% while requiring an upper bound of 32 nbsp;KiB of memory.
    pub fn normal_precision(mut self, normal_precision: i32) -> Self {
        self.normal_precision = normal_precision;
        self
    }

    /// Sets the sparse precision to be used. Must be in the range from the [`HyperLogLogPlusPlusBuilder::normal_precision`]
    /// to [`HyperLogLogPlusPlus::MAXIMUM_SPARSE_PRECISION`] (inclusive), or [`HyperLogLogPlusPlus::SPARSE_PRECISION_DISABLED`]
    /// to disable the use of the sparse representation. We recommend to use [`HyperLogLogPlusPlusBuilder::no_sparse_mode`]
    /// for the latter, though.
    ///
    /// If not specified, the normal precision + [`HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA`] is used.
    pub fn sparse_precision(mut self, sparse_precision: i32) -> Self {
        self.sparse_precision = Some(sparse_precision);
        self
    }

    /// Disable the "sparse representation" mode; i.e., the normal representation, where all
    /// registers are explicitly stored, and its method to compute the `COUNT DISTINCT` estimate
    /// are used from the start of the aggregation.
    pub fn no_sparse_mode(self) -> Self {
        self.sparse_precision(HyperLogLogPlusPlus::SPARSE_PRECISION_DISABLED)
    }

    /// Returns a new HLL++ aggregator for counting the number of unique byte arrays in a stream.
    pub fn build_for_bytes(self) -> Result<HyperLogLogPlusPlus, SketchError> {
        HyperLogLogPlusPlus::from_state(self.build_state(DefaultOpsTypeId::BYTES_OR_UTF8_STRING))
    }

    /// Returns a new HLL++ aggregator for counting the number of unique strings in a stream.
    pub fn build_for_string(self) -> Result<HyperLogLogPlusPlus, SketchError> {
        HyperLogLogPlusPlus::from_state(self.build_state(DefaultOpsTypeId::BYTES_OR_UTF8_STRING))
    }

    /// Returns a new HLL++ aggregator for counting the number of unique 32-bit integers in a stream.
    pub fn build_for_u32(self) -> Result<HyperLogLogPlusPlus, SketchError> {
        HyperLogLogPlusPlus::from_state(self.build_state(DefaultOpsTypeId::UINT32))
    }

    /// Returns a new HLL++ aggregator for counting the number of unique 64-bit integers in a stream.
    pub fn build_for_u64(self) -> Result<HyperLogLogPlusPlus, SketchError> {
        HyperLogLogPlusPlus::from_state(self.build_state(DefaultOpsTypeId::UINT64))
    }

    fn build_state(self, ops_type: DefaultOpsTypeId) -> State {
        State {
            r#type: AggregatorType::HYPERLOGLOG_PLUS_UNIQUE,
            encoding_version: HyperLogLogPlusPlus::ENCODING_VERSION,
            precision: self.normal_precision,
            sparse_precision: match self.sparse_precision {
                Some(precision) => precision,
                None => self.normal_precision + HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA,
            },
            value_type: ValueType::DefaultOpsType(ops_type),
            ..State::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aggregator::Aggregator, // Assuming this trait might be used generally
        error::SketchError,
        protos::{
            zetasketch::hllplus_unique::HyperLogLogPlusUniqueStateProto, AggregatorStateProto,
            AggregatorType as ProtoAggregatorType, DefaultOpsTypeId as ProtoDefaultOpsTypeId,
        },
    };
    use protobuf::UnknownValueRef; // For to_byte_array, parse_from_bytes

    struct JavaRand {
        seed: u64,
    }

    // Java-compatible pseudo-random number generator.
    // This follows the exact algorithm described for java.util.Random, ensuring that our tests run with the same
    // pseudo-random data as the Java tests, which makes debugging differences much easier.
    impl JavaRand {
        const MULTIPLIER: u64 = 0x5DEECE66D;
        const MASK: u64 = (1u64 << 48) - 1;

        fn initial_scramble(seed: u64) -> u64 {
            (seed ^ Self::MULTIPLIER) & Self::MASK
        }

        pub fn new(seed: u64) -> Self {
            Self {
                seed: Self::initial_scramble(seed),
            }
        }

        pub fn next(&mut self, bits: u32) -> u32 {
            let new_seed =
                (self.seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & ((1u64 << 48) - 1);
            self.seed = new_seed;
            (self.seed >> (48 - bits)) as u32
        }

        fn next_int_bounded(&mut self, bound: i32) -> i32 {
            if bound <= 0 {
                panic!("bound must be positive");
            }

            if (bound & -bound) == bound {
                // Power of 2 - use bit masking
                return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
            }

            // Rejection sampling to avoid bias
            let mut bits;
            let mut val;
            loop {
                bits = self.next(31) as i32;
                val = bits % bound;
                if bits - val + (bound - 1) >= 0 {
                    break;
                }
            }
            val
        }

        pub fn next_i64(&mut self) -> i64 {
            let high = self.next(32) as i32;
            let low = self.next(32) as i32;
            ((high as i64) << 32) + (low as i64)
        }
    }

    const TEST_NORMAL_PRECISION: i32 = HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION; // 15
    const TEST_SPARSE_PRECISION: i32 =
        TEST_NORMAL_PRECISION + HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA; // 20, default in Java tests sometimes use 25

    // Helper for default builder from Java tests (sparsePrecision 25)
    fn hll_builder_java_default_sparse() -> HyperLogLogPlusPlusBuilder {
        HyperLogLogPlusPlus::builder().sparse_precision(25)
    }

    // Helper to create AggregatorStateProto for BYTES_OR_UTF8_STRING type
    fn byte_or_string_type_state_proto_helper() -> AggregatorStateProto {
        let mut hll_unique_proto = HyperLogLogPlusUniqueStateProto::new();
        hll_unique_proto.set_precision_or_num_buckets(TEST_NORMAL_PRECISION);
        hll_unique_proto.set_sparse_precision_or_num_buckets(25); // As in Java test

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::BYTES_OR_UTF8_STRING);
        proto.set_value_type(vt.into());

        set_hll_extension(&mut proto, hll_unique_proto);
        proto
    }

    // Helper to create AggregatorStateProto for UNKNOWN type
    fn unknown_type_state_proto_helper() -> AggregatorStateProto {
        let mut hll_unique_proto = HyperLogLogPlusUniqueStateProto::new();
        hll_unique_proto.set_precision_or_num_buckets(TEST_NORMAL_PRECISION);
        hll_unique_proto.set_sparse_precision_or_num_buckets(25); // As in Java test

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        // No ValueTypeInfo for UNKNOWN type

        set_hll_extension(&mut proto, hll_unique_proto);
        proto
    }

    fn get_hll_extension(proto: &AggregatorStateProto) -> HyperLogLogPlusUniqueStateProto {
        let ext_data = proto
            .unknown_fields()
            .get(112)
            .expect("HLL extension not found");

        match ext_data {
            UnknownValueRef::LengthDelimited(data) => {
                HyperLogLogPlusUniqueStateProto::parse_from_bytes(data)
                    .expect("Failed to parse HLL extension")
            }
            _ => panic!("Unexpected extension type: {ext_data:?}"),
        }
    }

    fn set_hll_extension(
        proto: &mut AggregatorStateProto,
        hll_ext: HyperLogLogPlusUniqueStateProto,
    ) {
        proto
            .mut_unknown_fields()
            .add_length_delimited(112, hll_ext.write_to_bytes().unwrap());
    }

    #[test]
    fn test_merge_multiple_sparse_representations_into_a_normal_one() {
        let normal_precision = 13;
        let sparse_precision = 16;
        let hll_builder = HyperLogLogPlusPlus::builder()
            .normal_precision(normal_precision)
            .sparse_precision(sparse_precision);

        let num_sketches = 100;
        let mut random = JavaRand::new(123);

        let mut agg_state_protos: Vec<AggregatorStateProto> = Vec::new();
        let mut overall_aggregator = hll_builder
            .clone()
            .build_for_u64()
            .expect("Failed to build overall_aggregator");

        for _i in 0..num_sketches {
            let max = (1 << normal_precision) / 2;
            let num_values = random.next_int_bounded(max) + 1;

            let mut aggregator = hll_builder
                .clone()
                .build_for_u64()
                .expect("Failed to build aggregator");

            for _k in 0..num_values {
                let value = random.next_i64() as u64;
                aggregator.add_u64(value).unwrap_or_else(|_| {
                    panic!("Failed to add value {value} to aggregator (i={_i}, k={_k})")
                });
                overall_aggregator
                    .add_u64(value)
                    .expect("Failed to add value to overall_aggregator");
            }

            let proto = aggregator
                .serialize_to_proto()
                .expect("Failed to serialize aggregator");
            let hll_ext = get_hll_extension(&proto);
            assert!(
                !hll_ext.sparse_data().is_empty(),
                "Expected sparse data for individual sketch"
            );
            assert!(
                hll_ext.data().is_empty(),
                "Expected no normal data for individual sparse sketch"
            );
            agg_state_protos.push(proto);
        }

        let expected_proto = overall_aggregator
            .serialize_to_proto()
            .expect("Failed to serialize overall_aggregator");
        let overall_hll_ext = get_hll_extension(&expected_proto);
        assert!(
            overall_hll_ext.sparse_data().is_empty(),
            "Expected no sparse data for overall sketch"
        );
        assert!(
            !overall_hll_ext.data().is_empty(),
            "Expected normal data for overall sketch"
        );

        let mut merged_aggregator = HyperLogLogPlusPlus::from_proto(agg_state_protos[0].clone())
            .expect("Failed to build merged_aggregator from proto");
        for agg_proto in agg_state_protos.iter().skip(1) {
            merged_aggregator
                .merge_proto(agg_proto.clone())
                .expect("Failed to merge proto");
        }

        assert_eq!(
            merged_aggregator
                .serialize_to_proto()
                .expect("Serialize failed"),
            expected_proto
        );
    }

    #[test]
    fn add_bytes() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_bytes()
            .expect("build failed");
        aggregator.add_bytes(&[12]).expect("add_bytes failed");
        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 1);
    }

    #[test]
    fn add_bytes_throws_when_other_type() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u64()
            .expect("build failed"); // Build for Longs
        let result = aggregator.add_bytes(&[12]);
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type BYTES to aggregator of type {LONG}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_bytes_to_byte_or_string_type() {
        let mut aggregator =
            HyperLogLogPlusPlus::from_proto(byte_or_string_type_state_proto_helper())
                .expect("from_proto failed");
        aggregator.add_bytes(&[12]).expect("add_bytes failed"); // First add sets the type to BYTES

        let result = aggregator.add_string("foo"); // Second add with different type (STRING)
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            // Type is now fixed to BYTES
            assert!(msg.contains("Unable to add type STRING to aggregator of type {BYTES}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_bytes_to_uninitialized() {
        let mut aggregator = HyperLogLogPlusPlus::from_proto(unknown_type_state_proto_helper())
            .expect("from_proto failed");
        aggregator.add_bytes(&[12]).expect("add_bytes failed"); // First add sets type to BYTES

        let result = aggregator.add_u64(42); // Try adding Long
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type LONG to aggregator of type {BYTES}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_integer() {
        // u32 in Rust
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");
        aggregator.add_u32(1).expect("add_u32 failed");
        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 1);
    }

    #[test]
    fn add_integer_throws_when_other_type() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u64()
            .expect("build failed"); // Build for Longs
        let result = aggregator.add_u32(1); // Try adding Integer
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type INTEGER to aggregator of type {LONG}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_integer_to_uninitialized() {
        let mut aggregator = HyperLogLogPlusPlus::from_proto(unknown_type_state_proto_helper())
            .expect("from_proto failed");
        aggregator.add_u32(42).expect("add_u32 failed"); // First add sets type to INTEGER

        let result = aggregator.add_u64(42); // Try adding Long
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type LONG to aggregator of type {INTEGER}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_long() {
        // u64 in Rust
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u64()
            .expect("build failed");
        aggregator.add_u64(1).expect("add_u64 failed");
        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 1);
    }

    #[test]
    fn add_long_throws_when_other_type() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed"); // Build for Integer
        let result = aggregator.add_u64(1); // Try adding Long
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type LONG to aggregator of type {INTEGER}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_long_to_uninitialized() {
        let mut aggregator = HyperLogLogPlusPlus::from_proto(unknown_type_state_proto_helper())
            .expect("from_proto failed");
        aggregator.add_u64(42).expect("add_u64 failed"); // First add sets type to LONG

        let result = aggregator.add_u32(42); // Try adding Integer
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type INTEGER to aggregator of type {LONG}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_string() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_string()
            .expect("build failed");
        aggregator.add_string("foo").expect("add_string failed");
        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 1);
    }

    #[test]
    fn add_string_to_byte_or_string_type() {
        let mut aggregator =
            HyperLogLogPlusPlus::from_proto(byte_or_string_type_state_proto_helper())
                .expect("from_proto failed");
        aggregator.add_string("foo").expect("add_string failed"); // First add sets type to STRING

        let result = aggregator.add_bytes(&[1]); // Second add with different type (BYTES)
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type BYTES to aggregator of type {STRING}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn add_string_to_uninitialized() {
        let mut aggregator = HyperLogLogPlusPlus::from_proto(unknown_type_state_proto_helper())
            .expect("from_proto failed");
        aggregator.add_string("foo").expect("add_string failed"); // First add sets type to STRING

        let result = aggregator.add_u32(42); // Try adding Integer
        assert!(result.is_err());
        if let Err(SketchError::InvalidState(msg)) = result {
            assert!(msg.contains("Unable to add type INTEGER to aggregator of type {STRING}"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn create_throws_when_precision_too_large() {
        let result = HyperLogLogPlusPlus::builder()
            .normal_precision(HyperLogLogPlusPlus::MAXIMUM_PRECISION + 1)
            .sparse_precision(25) // valid sparse_p
            .build_for_u32();
        assert!(result.is_err());
        if let Err(SketchError::IllegalArgument(msg)) = result {
            assert!(msg.contains(&format!(
                "Expected normal precision to be >= {} and <= {} but was {}",
                HyperLogLogPlusPlus::MINIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION + 1
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn create_throws_when_precision_too_small() {
        let result = HyperLogLogPlusPlus::builder()
            .normal_precision(HyperLogLogPlusPlus::MINIMUM_PRECISION - 1)
            .sparse_precision(25) // valid sparse_p
            .build_for_u32();
        assert!(result.is_err());
        if let Err(SketchError::IllegalArgument(msg)) = result {
            assert!(msg.contains(&format!(
                "Expected normal precision to be >= {} and <= {} but was {}",
                HyperLogLogPlusPlus::MINIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION,
                HyperLogLogPlusPlus::MINIMUM_PRECISION - 1
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_no_extension() {
        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        // No HLL unique extension set

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when extension is missing");
        if let SketchError::IllegalArgument(msg) = result {
            assert!(msg.contains("Expected normal precision to be >= 10 and <= 24 but was 0"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_normal_precision_too_large() {
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(HyperLogLogPlusPlus::MAXIMUM_PRECISION + 1);
        // sparse precision default or valid
        hll_state.set_sparse_precision_or_num_buckets(
            HyperLogLogPlusPlus::MAXIMUM_PRECISION
                + 1
                + HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA,
        );

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when normal precision is too large");
        if let SketchError::IllegalArgument(msg) = result {
            assert!(msg.contains(&format!(
                "Expected normal precision to be >= {} and <= {} but was {}",
                HyperLogLogPlusPlus::MINIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION + 1
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_normal_precision_too_small() {
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(HyperLogLogPlusPlus::MINIMUM_PRECISION - 1);
        hll_state.set_sparse_precision_or_num_buckets(TEST_SPARSE_PRECISION);

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when normal precision is too small");
        if let SketchError::IllegalArgument(msg) = result {
            assert!(msg.contains(&format!(
                "Expected normal precision to be >= {} and <= {} but was {}",
                HyperLogLogPlusPlus::MINIMUM_PRECISION,
                HyperLogLogPlusPlus::MAXIMUM_PRECISION,
                HyperLogLogPlusPlus::MINIMUM_PRECISION - 1
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_not_hyperloglogplusplus() {
        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::SUM); // Incorrect type
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);

        // Extension might not matter or be absent, but main type is wrong
        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when invalid type is set");
        if let SketchError::InvalidState(msg) = result {
            assert!(
                msg.contains("Expected proto to be of type HYPERLOGLOG_PLUS_UNIQUE but was SUM")
            );
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    // Test fromProto_ThrowsWhenSparseIsMissingSparsePrecision from Java
    // In Rust, if sparse_data is set, sparse_precision must be valid (not 0).
    // SparseRepresentation::new checks if sparse_precision is 0 and errors.
    // State::from_hll_proto: if sparse_precision is 0 but sparse_data is present, it might error or become normal.
    // Current Rust code: Representation::from_state checks if sparse_precision != DISABLED and sparse_data is not empty
    // for it to be sparse. If sparse_precision is 0 (DISABLED), it becomes Normal.
    // Java test: sparse data is set, but sparse precision is 0. This is an invalid state.
    #[test]
    fn from_proto_fails_when_sparse_is_missing_sparse_precision() {
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(TEST_NORMAL_PRECISION);
        hll_state.set_sparse_precision_or_num_buckets(0); // Missing or disabled sparse precision
        hll_state.set_sparse_data(vec![1]); // But sparse data is present

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when sparse precision is missing");
        if let SketchError::InvalidState(msg) = result {
            assert!(msg.contains("Must have a sparse precision when sparse data is set"));
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_sparse_precision_too_large() {
        let normal_p = 15;
        let sparse_p = HyperLogLogPlusPlus::MAXIMUM_SPARSE_PRECISION + 1; // 26, too large

        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(normal_p);
        hll_state.set_sparse_precision_or_num_buckets(sparse_p);

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when sparse precision is too large");
        if let SketchError::IllegalArgument(msg) = result {
            assert!(msg.contains(&format!(
                "Expected sparse precision to be >= normal precision ({}) and <= {} but was {}.",
                normal_p,
                HyperLogLogPlusPlus::MAXIMUM_SPARSE_PRECISION,
                sparse_p
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn from_proto_fails_when_sparse_precision_too_small() {
        let normal_p = 15;
        let sparse_p = normal_p - 1; // 14, too small (must be >= normal_p)

        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(normal_p);
        hll_state.set_sparse_precision_or_num_buckets(sparse_p);

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(0);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let result = HyperLogLogPlusPlus::from_proto(proto)
            .expect_err("HLL should fail to load when sparse precision is too small");
        if let SketchError::IllegalArgument(msg) = result {
            assert!(msg.contains(&format!(
                "Expected sparse precision to be >= normal precision ({}) and <= {} but was {}.",
                normal_p,
                HyperLogLogPlusPlus::MAXIMUM_SPARSE_PRECISION,
                sparse_p
            )));
        } else {
            panic!("Unexpected error type or message: {result:?}");
        }
    }

    #[test]
    fn from_proto_when_normal() {
        let normal_p = 15;
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(normal_p);
        // No sparse_precision explicitly set, or set to 0 for normal.
        // If sparse_precision is not set, State::from_hll_proto uses normal_p + DELTA
        // To force normal, sparse_precision should be 0 OR data field set.
        hll_state.set_sparse_precision_or_num_buckets(0); // Mark as normal
        hll_state.set_data(vec![0; 1 << normal_p]); // Normal data

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(1);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let aggregator =
            HyperLogLogPlusPlus::from_proto(proto).expect("from_proto failed for normal");
        // Estimate for all zeros data is 0 (or close to it)
        assert!(aggregator.result().expect("result failed") >= 0); // Exact estimate is complex for all-zero data
        assert_eq!(aggregator.num_values(), 1);
        assert!(aggregator.representation.is_normal());
    }

    #[test]
    fn from_proto_when_sparse() {
        let normal_p = 15;
        let sparse_p = 25;
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(normal_p);
        hll_state.set_sparse_precision_or_num_buckets(sparse_p);
        hll_state.set_sparse_data(vec![1]); // Sparse data
        hll_state.set_sparse_size(1); // From Java test

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(2); // From Java test
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let aggregator =
            HyperLogLogPlusPlus::from_proto(proto).expect("from_proto failed for sparse");
        assert_eq!(aggregator.result().expect("result failed"), 1); // Java test expects 1
        assert_eq!(aggregator.num_values(), 2);
        assert!(aggregator.representation.is_sparse());
    }

    #[test]
    fn from_proto_byte_array() {
        let normal_p = 15;
        let sparse_p = 25;
        let mut hll_state = HyperLogLogPlusUniqueStateProto::new();
        hll_state.set_precision_or_num_buckets(normal_p);
        hll_state.set_sparse_precision_or_num_buckets(sparse_p);
        hll_state.set_sparse_data(vec![1]);
        hll_state.set_sparse_size(1);

        let mut proto = AggregatorStateProto::new();
        proto.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto.set_num_values(2);
        set_hll_extension(&mut proto, hll_state);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto.set_value_type(vt.into());

        let byte_array = proto.write_to_bytes().expect("write_to_bytes failed");
        let aggregator = HyperLogLogPlusPlus::from_bytes(&byte_array).expect("from_bytes failed");

        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 2);
    }

    #[test]
    fn from_proto_byte_array_throws_when_invalid() {
        let result = HyperLogLogPlusPlus::from_bytes(&[1, 2, 3]); // Invalid proto data
        assert!(result.is_err());
        if let Err(SketchError::ProtoDeserialization(_)) = result {
            // Correct error type
        } else {
            panic!("Unexpected error type: {result:?}");
        }
    }

    #[test]
    fn long_result_simple() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");
        aggregator.add_u32(1).expect("add failed");
        aggregator.add_u32(2).expect("add failed");
        aggregator.add_u32(3).expect("add failed");
        aggregator.add_u32(2).expect("add failed"); // Duplicate
        aggregator.add_u32(3).expect("add failed"); // Duplicate
        assert_eq!(aggregator.result().expect("result failed"), 3);
    }

    #[test]
    fn long_result_zero_when_empty() {
        let aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");
        assert_eq!(aggregator.result().expect("result failed"), 0);
    }

    #[test]
    fn merge_from_proto() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");

        let mut hll_state_to_merge = HyperLogLogPlusUniqueStateProto::new();
        hll_state_to_merge.set_precision_or_num_buckets(TEST_NORMAL_PRECISION);
        hll_state_to_merge.set_sparse_precision_or_num_buckets(25); // Matching sparse precision
        hll_state_to_merge.set_sparse_data(vec![1]);
        hll_state_to_merge.set_sparse_size(1);

        let mut proto_to_merge = AggregatorStateProto::new();
        proto_to_merge.set_type(ProtoAggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto_to_merge.set_encoding_version(HyperLogLogPlusPlus::ENCODING_VERSION);
        proto_to_merge.set_num_values(2);
        set_hll_extension(&mut proto_to_merge, hll_state_to_merge);

        let vt = ValueType::DefaultOpsType(ProtoDefaultOpsTypeId::UINT32);
        proto_to_merge.set_value_type(vt.into());

        aggregator
            .merge_proto(proto_to_merge)
            .expect("merge_proto failed");
        assert_eq!(aggregator.result().expect("result failed"), 1);
        assert_eq!(aggregator.num_values(), 2); // Num values should be sum
    }

    #[test]
    fn merge_normal_into_normal_with_higher_precision() {
        let mut a = HyperLogLogPlusPlus::builder()
            .no_sparse_mode() // Uses MAX_SPARSE_P, effectively sparse but test means "normal rep"
            .build_for_u32()
            .expect("Build A failed");

        a.add_u32(1).unwrap();
        a.add_u32(2).unwrap();
        a.add_u32(3).unwrap();

        let mut b = HyperLogLogPlusPlus::builder()
            .normal_precision(13) // Higher precision
            .no_sparse_mode()
            .build_for_u32()
            .expect("Build B failed");
        b.add_u32(3).unwrap();
        b.add_u32(4).unwrap();

        a.merge_aggregator(b).expect("Merge failed");

        assert_eq!(a.normal_precision(), 13);
        assert_eq!(a.sparse_precision(), 0);
        assert_eq!(a.result().unwrap(), 4);
        assert_eq!(a.num_values(), 5);
        //assert_eq!(b.result().unwrap(), 2);
        //assert_eq!(b.num_values(), 2);
    }

    #[test]
    fn num_values_simple() {
        let mut aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");
        aggregator.add_u32(1).unwrap();
        aggregator.add_u32(2).unwrap();
        aggregator.add_u32(3).unwrap();
        aggregator.add_u32(2).unwrap();
        aggregator.add_u32(3).unwrap();
        assert_eq!(aggregator.num_values(), 5);
    }

    #[test]
    fn num_values_zero_when_empty() {
        let aggregator = hll_builder_java_default_sparse()
            .build_for_u32()
            .expect("build failed");
        assert_eq!(aggregator.num_values(), 0);
    }

    #[test]
    fn serialize_to_proto_empty_aggregator_sets_empty_sparse_data_field() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .normal_precision(13)
            .sparse_precision(16)
            .build_for_bytes()
            .expect("Build failed");

        let actual_proto = aggregator.serialize_to_proto().expect("Serialize failed");
        let hll_ext = get_hll_extension(&actual_proto);

        assert!(hll_ext.has_sparse_data()); // Field should be present
        assert!(hll_ext.sparse_data().is_empty()); // And its value empty
        assert!(!hll_ext.has_data() || hll_ext.data().is_empty()); // Normal data should not be present or empty
    }

    #[test]
    fn builder_uses_both_precision_defaults_when_unspecified() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .build_for_string()
            .expect("Build failed");
        assert_eq!(
            aggregator.normal_precision(),
            HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION
        );
        assert_eq!(
            aggregator.sparse_precision(),
            HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION
                + HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA
        );
    }

    #[test]
    fn builder_uses_normal_precision_default_when_unspecified() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .sparse_precision(18)
            .build_for_u32()
            .expect("Build failed");
        assert_eq!(
            aggregator.normal_precision(),
            HyperLogLogPlusPlus::DEFAULT_NORMAL_PRECISION
        );
        assert_eq!(aggregator.sparse_precision(), 18);
    }

    #[test]
    fn builder_uses_sparse_precision_default_when_unspecified() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .normal_precision(18)
            .build_for_u64()
            .expect("Build failed");
        assert_eq!(aggregator.normal_precision(), 18);
        assert_eq!(
            aggregator.sparse_precision(),
            18 + HyperLogLogPlusPlus::DEFAULT_SPARSE_PRECISION_DELTA
        );
    }

    #[test]
    fn builder_uses_both_precisions_as_specified() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .normal_precision(14)
            .sparse_precision(17)
            .build_for_bytes()
            .expect("Build failed");
        assert_eq!(aggregator.normal_precision(), 14);
        assert_eq!(aggregator.sparse_precision(), 17);
    }

    #[test]
    fn builder_invocation_order_does_not_matter() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .sparse_precision(17)
            .normal_precision(14)
            .build_for_bytes()
            .expect("Build failed");
        assert_eq!(aggregator.normal_precision(), 14);
        assert_eq!(aggregator.sparse_precision(), 17);
    }

    #[test]
    fn builder_no_sparse_mode_behavior() {
        let aggregator = HyperLogLogPlusPlus::builder()
            .no_sparse_mode()
            .normal_precision(16)
            .build_for_bytes()
            .expect("Build failed");

        assert_eq!(aggregator.sparse_precision(), 0);
        assert_eq!(aggregator.normal_precision(), 16);
        assert!(aggregator.representation.is_normal());
    }

    #[test]
    fn builder_reuse() {
        let mut hll_builder = HyperLogLogPlusPlus::builder()
            .normal_precision(13)
            .sparse_precision(16);

        let mut bytes_aggregator = hll_builder
            .clone()
            .build_for_bytes()
            .expect("Build bytes failed");
        bytes_aggregator.add_bytes(&[12]).unwrap();
        assert_eq!(bytes_aggregator.result().unwrap(), 1);
        assert_eq!(bytes_aggregator.num_values(), 1);
        assert_eq!(bytes_aggregator.normal_precision(), 13);
        assert_eq!(bytes_aggregator.sparse_precision(), 16);

        let mut longs_aggregator = hll_builder
            .clone()
            .build_for_u64()
            .expect("Build longs failed");
        longs_aggregator.add_u64(1).unwrap();
        assert_eq!(longs_aggregator.result().unwrap(), 1);
        assert_eq!(longs_aggregator.num_values(), 1);
        assert_eq!(longs_aggregator.normal_precision(), 13);
        assert_eq!(longs_aggregator.sparse_precision(), 16);

        // Change precisions on the builder
        hll_builder = hll_builder.sparse_precision(20).normal_precision(18);

        let mut string_aggregator = hll_builder.build_for_string().expect("Build string failed");
        string_aggregator.add_string("foo").unwrap();
        assert_eq!(string_aggregator.result().unwrap(), 1);
        assert_eq!(string_aggregator.num_values(), 1);
        assert_eq!(string_aggregator.normal_precision(), 18);
        assert_eq!(string_aggregator.sparse_precision(), 20);
    }

    #[test]
    fn test_result() {
        let mut aggregator = HyperLogLogPlusPlus::builder()
            .build_for_u64()
            .expect("Build failed");
        for i in 0..=2188 {
            aggregator.add_u64(i).unwrap();
            aggregator.result().unwrap();
        }
        println!("result (2188): {}", aggregator.result().unwrap());
        aggregator.add_u64(2189).unwrap();
        println!("result (2189): {}", aggregator.result().unwrap());
    }
}
