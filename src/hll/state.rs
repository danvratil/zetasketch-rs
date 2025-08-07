// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

// Replicates com.google.zetasketch.internal.hllplus.State.java
// This struct holds the serializable state of the HLL++ sketch.

use crate::error::SketchError;
use crate::hll::value_type::ValueType;
use crate::protos::{AggregatorStateProto, AggregatorType, HyperLogLogPlusUniqueStateProto};
use protobuf::rt::WireType;
use protobuf::{CodedInputStream, CodedOutputStream, Enum};

#[derive(Debug, Clone)]
pub struct State {
    /// The type of the aggregator
    pub r#type: AggregatorType,
    /// The number of values the aggregator has seen
    pub num_values: i64,
    /// Version of the encoded internal state
    pub encoding_version: i32,
    /// The value type for the aggregation
    pub value_type: ValueType,
    /// Size of the sparse list
    pub sparse_size: i32,
    /// Precision / number of buckets for the normal representation
    pub precision: i32,
    /// Precision / number of buckets for the sparse representation
    pub sparse_precision: i32,
    /// Normal data representation
    pub data: Option<Vec<u8>>,
    /// Sparse data representation
    pub sparse_data: Option<Vec<u8>>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            r#type: Self::DEFAULT_TYPE,
            num_values: Self::DEFAULT_NUM_VALUES,
            encoding_version: Self::DEFAULT_ENCODING_VERSION,
            value_type: ValueType::Unknown,
            sparse_size: Self::DEFAULT_SPARSE_SIZE,
            precision: Self::DEFAULT_PRECISION_OR_NUM_BUCKETS,
            sparse_precision: Self::DEFAULT_SPARSE_PRECISION_OR_NUM_BUCKETS,
            data: None,
            sparse_data: None,
        }
    }
}

impl State {
    // The numbers are field numbers from the AggregatorStateProto message.
    // Unfortunately, Rust's protobuf codegen doesn't generate field number constants for us,
    // so I just had to copy them over manually
    const TYPE_TAG: u32 = 1 << 3 | (WireType::Varint as u32);
    const NUM_VALUES_TAG: u32 = 2 << 3 | (WireType::Varint as u32);
    const ENCODING_VERSION_TAG: u32 = 3 << 3 | (WireType::Varint as u32);
    const VALUE_TYPE_TAG: u32 = 4 << 3 | (WireType::Varint as u32);
    const HYPERLOGLOGPLUS_UNIQUE_STATE_TAG: u32 = 112 << 3 | (WireType::LengthDelimited as u32);

    const SPARSE_SIZE_TAG: u32 = 2 << 3 | (WireType::Varint as u32);
    const PRECISION_OR_NUM_BUCKETS_TAG: u32 = 3 << 3 | (WireType::Varint as u32);
    const SPARSE_PRECISION_OR_NUM_BUCKETS_TAG: u32 = 4 << 3 | (WireType::Varint as u32);
    const DATA_TAG: u32 = 5 << 3 | (WireType::LengthDelimited as u32);
    const SPARSE_DATA_TAG: u32 = 6 << 3 | (WireType::LengthDelimited as u32);

    const DEFAULT_TYPE: AggregatorType = AggregatorType::HYPERLOGLOG_PLUS_UNIQUE;
    const DEFAULT_NUM_VALUES: i64 = 0;
    const DEFAULT_ENCODING_VERSION: i32 = 1;
    const DEFAULT_VALUE_TYPE: i32 = 0; // 0 is always UNKNOWN

    const DEFAULT_SPARSE_SIZE: i32 = 0;
    const DEFAULT_PRECISION_OR_NUM_BUCKETS: i32 = 0;
    const DEFAULT_SPARSE_PRECISION_OR_NUM_BUCKETS: i32 = 0;

    #[cfg(test)]
    fn has_data(&self) -> bool {
        match &self.data {
            Some(data) => !data.is_empty(),
            None => false,
        }
    }

    pub fn has_sparse_data(&self) -> bool {
        match &self.sparse_data {
            Some(data) => !data.is_empty(),
            None => false,
        }
    }

    fn parse_hll(
        stream: &mut CodedInputStream,
        size: i32,
    ) -> Result<HyperLogLogPlusUniqueStateProto, SketchError> {
        let limit = stream.pos() + size as u64;
        let mut hll = HyperLogLogPlusUniqueStateProto::default();
        while stream.pos() < limit && !stream.eof().map_err(SketchError::ProtoDeserialization)? {
            match stream.read_raw_tag_or_eof() {
                Ok(Some(Self::SPARSE_SIZE_TAG)) => {
                    hll.sparse_size = Some(
                        stream
                            .read_int32()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::PRECISION_OR_NUM_BUCKETS_TAG)) => {
                    hll.precision_or_num_buckets = Some(
                        stream
                            .read_int32()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::SPARSE_PRECISION_OR_NUM_BUCKETS_TAG)) => {
                    hll.sparse_precision_or_num_buckets = Some(
                        stream
                            .read_int32()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::DATA_TAG)) => {
                    hll.data = Some(
                        stream
                            .read_bytes()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::SPARSE_DATA_TAG)) => {
                    hll.sparse_data = Some(
                        stream
                            .read_bytes()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(tag)) => {
                    protobuf::rt::skip_field_for_tag(tag, stream)
                        .map_err(SketchError::ProtoDeserialization)?;
                }
                Ok(None) => break, // EOF
                Err(e) => {
                    return Err(SketchError::ProtoDeserialization(e));
                }
            }
        }
        Ok(hll)
    }

    /// Parses from a serialized AggregatorStateProto.
    pub fn parse(input: &[u8]) -> Result<Self, SketchError> {
        let mut stream = CodedInputStream::from_bytes(input);

        let mut state = AggregatorStateProto::new();
        let mut hll: Option<HyperLogLogPlusUniqueStateProto> = None;
        // We are using a manual parsing loop here because the generated Protobuf code cannot
        // handle extensions (or at least I don't know how to do it). The HLL++ state is an extension
        // of the AggregatorStateProto, so I figure the best way is to parse the protobuf by hand.
        loop {
            match stream.read_raw_tag_or_eof() {
                Ok(Some(Self::TYPE_TAG)) => {
                    state.type_ = Some(
                        stream
                            .read_enum_or_unknown::<AggregatorType>()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::NUM_VALUES_TAG)) => {
                    state.num_values = Some(
                        stream
                            .read_int64()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::ENCODING_VERSION_TAG)) => {
                    state.encoding_version = Some(
                        stream
                            .read_int32()
                            .map_err(SketchError::ProtoDeserialization)?,
                    );
                }
                Ok(Some(Self::VALUE_TYPE_TAG)) => {
                    state.value_type = Some(
                        stream
                            .read_int32()
                            .map_err(SketchError::ProtoDeserialization)?,
                    )
                }
                Ok(Some(Self::HYPERLOGLOGPLUS_UNIQUE_STATE_TAG)) => {
                    let size = stream
                        .read_int32()
                        .map_err(SketchError::ProtoDeserialization)?;
                    hll = Some(Self::parse_hll(&mut stream, size)?);
                }
                Ok(Some(tag)) => {
                    protobuf::rt::skip_field_for_tag(tag, &mut stream)
                        .map_err(SketchError::ProtoDeserialization)?;
                }
                Ok(None) => break, // EOF
                Err(e) => {
                    return Err(SketchError::ProtoDeserialization(e));
                }
            }
        }

        Ok(State {
            r#type: state.type_(),
            num_values: state.num_values(),
            encoding_version: state.encoding_version(),
            value_type: ValueType::from(state.value_type()),
            sparse_size: hll
                .as_ref()
                .map(|hll| hll.sparse_size())
                .unwrap_or(Self::DEFAULT_SPARSE_SIZE),
            precision: hll
                .as_ref()
                .map(|hll| hll.precision_or_num_buckets())
                .unwrap_or(Self::DEFAULT_PRECISION_OR_NUM_BUCKETS),
            sparse_precision: hll
                .as_ref()
                .map(|hll| hll.sparse_precision_or_num_buckets())
                .unwrap_or(Self::DEFAULT_SPARSE_PRECISION_OR_NUM_BUCKETS),
            data: hll.as_ref().and_then(|hll| hll.data.clone()),
            sparse_data: hll.as_ref().and_then(|hll| hll.sparse_data.clone()),
        })
    }

    /// Serializes to a byte array (AggregatorStateProto).
    pub fn to_byte_array(&self) -> Result<Vec<u8>, SketchError> {
        fn write_hll_to_buffer(state: &State, buffer: &mut Vec<u8>) -> Result<(), protobuf::Error> {
            let mut stream = CodedOutputStream::new(buffer);
            if state.sparse_size != State::DEFAULT_SPARSE_SIZE {
                stream.write_uint32_no_tag(State::SPARSE_SIZE_TAG)?;
                stream.write_int32_no_tag(state.sparse_size)?;
            }

            if state.precision != State::DEFAULT_PRECISION_OR_NUM_BUCKETS {
                stream.write_uint32_no_tag(State::PRECISION_OR_NUM_BUCKETS_TAG)?;
                stream.write_int32_no_tag(state.precision)?;
            }

            if state.sparse_precision != State::DEFAULT_SPARSE_PRECISION_OR_NUM_BUCKETS {
                stream.write_uint32_no_tag(State::SPARSE_PRECISION_OR_NUM_BUCKETS_TAG)?;
                stream.write_int32_no_tag(state.sparse_precision)?;
            }

            if let Some(data) = &state.data {
                stream.write_uint32_no_tag(State::DATA_TAG)?;
                stream.write_bytes_no_tag(data)?;
            }

            if let Some(data) = &state.sparse_data {
                stream.write_uint32_no_tag(State::SPARSE_DATA_TAG)?;
                stream.write_bytes_no_tag(data)?;
            }

            stream.flush()?;
            Ok(())
        }

        fn write_to_buffer(state: &State, buffer: &mut Vec<u8>) -> Result<(), protobuf::Error> {
            let mut stream = CodedOutputStream::new(buffer);

            stream.write_uint32_no_tag(State::TYPE_TAG)?;
            stream.write_enum_no_tag(state.r#type.value())?;

            stream.write_uint32_no_tag(State::NUM_VALUES_TAG)?;
            stream.write_int64_no_tag(state.num_values)?;

            if state.encoding_version != State::DEFAULT_ENCODING_VERSION {
                stream.write_uint32_no_tag(State::ENCODING_VERSION_TAG)?;
                stream.write_int32_no_tag(state.encoding_version)?;
            }

            if i32::from(state.value_type) != State::DEFAULT_VALUE_TYPE {
                stream.write_uint32_no_tag(State::VALUE_TYPE_TAG)?;
                stream.write_enum_no_tag(i32::from(state.value_type))?;
            }

            let mut hll_buffer = Vec::new();
            write_hll_to_buffer(state, &mut hll_buffer)?;
            stream.write_uint32_no_tag(State::HYPERLOGLOGPLUS_UNIQUE_STATE_TAG)?;
            stream.write_uint32_no_tag(hll_buffer.len() as u32)?;
            stream.write_raw_bytes(&hll_buffer)?;

            stream.flush()?;

            Ok(())
        }

        let mut buffer = Vec::new();
        write_to_buffer(self, &mut buffer).map_err(SketchError::ProtoSerialization)?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use protobuf::{EnumFull, Message, UnknownValueRef};

    use crate::protos::{exts, CustomValueTypeId, DefaultOpsTypeId};

    use super::*;

    fn aggregator_state_proto() -> AggregatorStateProto {
        let mut proto = AggregatorStateProto::default();
        proto.set_type(AggregatorType::HYPERLOGLOG_PLUS_UNIQUE);
        proto.set_num_values(0);
        proto
    }

    fn parse(proto: &AggregatorStateProto) -> State {
        State::parse(
            proto
                .write_to_bytes()
                .expect("Failed to serialize input message")
                .as_slice(),
        )
        .expect("Failed to parse input message")
    }

    fn parse_hll(proto: &HyperLogLogPlusUniqueStateProto) -> State {
        let mut buf = Vec::new();
        {
            let aggr_proto = aggregator_state_proto();
            let mut stream = CodedOutputStream::vec(&mut buf);
            aggr_proto
                .write_to(&mut stream)
                .expect("Failed to serialize input message");
            stream
                .write_uint32_no_tag(State::HYPERLOGLOGPLUS_UNIQUE_STATE_TAG)
                .expect("Failed to write tag");
            stream
                .write_message_no_tag(proto)
                .expect("Failed to serialize input message");
            stream.flush().expect("Failed to flush stream");
        }

        State::parse(&buf).expect("Failed to parse input message")
    }

    fn to_proto(state: &State) -> AggregatorStateProto {
        let buf = state.to_byte_array().expect("Failed to serialize state");
        AggregatorStateProto::parse_from_bytes(&buf).expect("Failed to parse input message")
    }

    fn to_hll_proto(state: &State) -> HyperLogLogPlusUniqueStateProto {
        let buf = state.to_byte_array().expect("Failed to serialize state");
        let proto =
            AggregatorStateProto::parse_from_bytes(&buf).expect("Failed to parse input message");
        exts::hyperloglogplus_unique_state
            .get(&proto)
            .unwrap_or_default()
    }

    #[test]
    fn test_has_data() {
        let mut state = State::default();
        assert!(!state.has_data());

        state.data = Some(vec![]);
        assert!(!state.has_data());

        state.data = Some(vec![1, 2, 3]);
        assert!(state.has_data());
    }

    #[test]
    fn test_has_sparse_data() {
        let mut state = State::default();
        assert!(!state.has_sparse_data());

        state.sparse_data = Some(vec![]);
        assert!(!state.has_sparse_data());

        state.sparse_data = Some(vec![1, 2, 3]);
        assert!(state.has_sparse_data());
    }

    #[test]
    fn test_parse_num_values() {
        let mut proto = aggregator_state_proto();

        proto.set_num_values(0);
        let state = parse(&proto);
        assert_eq!(state.num_values, 0);

        proto.num_values = Some(53);
        let state = parse(&proto);
        assert_eq!(state.num_values, 53);

        proto.num_values = Some(i64::MIN);
        let state = parse(&proto);
        assert_eq!(state.num_values, i64::MIN);

        proto.num_values = Some(i64::MAX);
        let state = parse(&proto);
        assert_eq!(state.num_values, i64::MAX);
    }

    #[test]
    fn test_parse_encoding_version() {
        let mut proto = aggregator_state_proto();

        let state = parse(&proto);
        assert_eq!(
            state.encoding_version, 1,
            "Default encoding version should be 1"
        );

        proto.set_encoding_version(0);
        let state = parse(&proto);
        assert_eq!(state.encoding_version, 0);

        proto.set_encoding_version(42);
        let state = parse(&proto);
        assert_eq!(state.encoding_version, 42);

        proto.set_encoding_version(i32::MIN);
        let state = parse(&proto);
        assert_eq!(state.encoding_version, i32::MIN);

        proto.set_encoding_version(i32::MAX);
        let state = parse(&proto);
        assert_eq!(state.encoding_version, i32::MAX);
    }

    #[test]
    fn test_parse_value_type_unknown() {
        let proto = aggregator_state_proto();

        let state = parse(&proto);
        assert_eq!(state.value_type, ValueType::Unknown);
    }

    #[test]
    fn test_parse_value_type_default_ops_type() {
        for enum_value in DefaultOpsTypeId::enum_descriptor().values() {
            let mut proto = aggregator_state_proto();
            proto.set_value_type(enum_value.value());

            let state = parse(&proto);
            let value_type = enum_value
                .cast::<DefaultOpsTypeId>()
                .expect("Failed to cast enum value");
            assert_eq!(state.value_type, ValueType::from(value_type.value()));
        }
    }

    #[test]
    fn test_parse_value_type_custom_value_type() {
        for enum_value in CustomValueTypeId::enum_descriptor().values() {
            let mut proto = aggregator_state_proto();
            proto.set_value_type(enum_value.value());

            let state = parse(&proto);
            let value_type = enum_value
                .cast::<CustomValueTypeId>()
                .expect("Failed to cast enum value");
            assert_eq!(state.value_type, ValueType::from(value_type.value()));
        }
    }

    #[test]
    fn test_parse_value_type_other_number() {
        let mut proto = aggregator_state_proto();
        proto.set_value_type(12345);

        let state = parse(&proto);
        assert_eq!(state.value_type, ValueType::CustomType(12345));
    }

    #[test]
    fn test_parse_sparse_size() {
        let mut proto = HyperLogLogPlusUniqueStateProto::default();

        proto.clear_sparse_size();
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_size, 0);

        proto.set_sparse_size(0);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_size, 0);

        proto.set_sparse_size(42);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_size, 42);

        proto.set_sparse_size(i32::MIN);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_size, i32::MIN);

        proto.set_sparse_size(i32::MAX);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_size, i32::MAX);
    }

    #[test]
    fn test_parse_precision() {
        let mut proto = HyperLogLogPlusUniqueStateProto::default();

        proto.clear_precision_or_num_buckets();
        let state = parse_hll(&proto);
        assert_eq!(state.precision, 0);

        proto.set_precision_or_num_buckets(0);
        let state = parse_hll(&proto);
        assert_eq!(state.precision, 0);

        proto.set_precision_or_num_buckets(42);
        let state = parse_hll(&proto);
        assert_eq!(state.precision, 42);

        proto.set_precision_or_num_buckets(i32::MIN);
        let state = parse_hll(&proto);
        assert_eq!(state.precision, i32::MIN);

        proto.set_precision_or_num_buckets(i32::MAX);
        let state = parse_hll(&proto);
        assert_eq!(state.precision, i32::MAX);
    }

    #[test]
    fn test_parse_sparse_precision() {
        let mut proto = HyperLogLogPlusUniqueStateProto::default();

        proto.clear_sparse_precision_or_num_buckets();
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_precision, 0);

        proto.set_sparse_precision_or_num_buckets(0);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_precision, 0);

        proto.set_sparse_precision_or_num_buckets(42);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_precision, 42);

        proto.set_sparse_precision_or_num_buckets(i32::MIN);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_precision, i32::MIN);

        proto.set_sparse_precision_or_num_buckets(i32::MAX);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_precision, i32::MAX);
    }

    #[test]
    fn test_parse_data() {
        let mut proto = HyperLogLogPlusUniqueStateProto::default();

        proto.clear_data();
        let state = parse_hll(&proto);
        assert!(state.data.is_none());

        proto.set_data(vec![]);
        let state = parse_hll(&proto);
        assert_eq!(state.data, Some(vec![]));

        proto.set_data(vec![1, 2, 3]);
        let state = parse_hll(&proto);
        assert_eq!(state.data, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_parse_sparse_data() {
        let mut proto = HyperLogLogPlusUniqueStateProto::default();

        proto.clear_sparse_data();
        let state = parse_hll(&proto);
        assert!(state.sparse_data.is_none());

        proto.set_sparse_data(vec![]);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_data, Some(vec![]));

        proto.set_sparse_data(vec![1, 2, 3]);
        let state = parse_hll(&proto);
        assert_eq!(state.sparse_data, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_parse_unknown_fields() {
        let mut buf = Vec::new();
        {
            let mut stream = CodedOutputStream::vec(&mut buf);
            stream
                .write_uint32_no_tag(State::NUM_VALUES_TAG)
                .expect("Failed to write tag");
            stream
                .write_int32_no_tag(42)
                .expect("Failed to write int32");
            stream
                .write_string(999, "foobar")
                .expect("Failed to write string");
            stream
                .write_uint32_no_tag(State::ENCODING_VERSION_TAG)
                .expect("Failed to write tag");
            stream
                .write_int32_no_tag(43)
                .expect("Failed to write int32");
            stream.flush().expect("Failed to flush stream");
        }

        // Check that we can parse the proto, despite the unknown field.
        let state = State::parse(&buf).expect("Failed to parse input message");
        assert_eq!(state.num_values, 42);
        assert_eq!(state.encoding_version, 43);
    }

    #[test]
    fn test_serialize_type() {
        let mut state = State::default();

        let actual = to_proto(&state);
        assert!(actual.has_type());
        assert_eq!(actual.type_(), AggregatorType::HYPERLOGLOG_PLUS_UNIQUE);

        for enum_value in AggregatorType::enum_descriptor().values() {
            state.r#type = enum_value
                .cast::<AggregatorType>()
                .expect("Failed to cast enum value");
            let actual = to_proto(&state);
            assert!(actual.has_type());
            assert_eq!(
                actual.type_(),
                enum_value
                    .cast::<AggregatorType>()
                    .expect("Failed to cast enum value")
            );
        }
    }

    #[test]
    fn test_serialize_num_values() {
        let mut state = State::default();

        let actual = to_proto(&state);
        assert!(actual.has_num_values());
        assert_eq!(actual.num_values(), 0);

        state.num_values = 42;
        let actual = to_proto(&state);
        assert!(actual.has_num_values());
        assert_eq!(actual.num_values(), 42);
    }

    #[test]
    fn test_serialize_encoding_version() {
        let mut state = State::default();

        let actual = to_proto(&state);
        assert!(!actual.has_encoding_version());

        state.encoding_version = 2;
        let actual = to_proto(&state);
        assert!(actual.has_encoding_version());
        assert_eq!(actual.encoding_version(), 2);
    }

    #[test]
    fn test_serialize_value_type() {
        let mut state = State::default();

        let actual = to_proto(&state);
        assert!(!actual.has_value_type());

        for enum_value in DefaultOpsTypeId::enum_descriptor().values() {
            let value_type = ValueType::DefaultOpsType(
                enum_value
                    .cast::<DefaultOpsTypeId>()
                    .expect("Failed to cast enum value"),
            );
            state.value_type = value_type;
            let actual = to_proto(&state);
            if value_type == ValueType::Unknown {
                assert!(!actual.has_value_type());
            }
            assert_eq!(actual.value_type(), enum_value.value());
        }
    }

    #[test]
    fn test_serialize_sparse_size() {
        let mut state = State::default();

        let proto = to_hll_proto(&state);
        assert!(!proto.has_sparse_size());

        state.sparse_size = 42;
        let proto = to_hll_proto(&state);
        assert!(proto.has_sparse_size());
        assert_eq!(proto.sparse_size(), 42);
    }

    #[test]
    fn test_serialize_precision() {
        let mut state = State::default();

        let proto = to_hll_proto(&state);
        assert!(!proto.has_precision_or_num_buckets());

        state.precision = 42;
        let proto = to_hll_proto(&state);
        assert!(proto.has_precision_or_num_buckets());
        assert_eq!(proto.precision_or_num_buckets(), 42);
    }

    #[test]
    fn test_serialize_sparse_precision() {
        let mut state = State::default();

        let proto = to_hll_proto(&state);
        assert!(!proto.has_sparse_precision_or_num_buckets());

        state.sparse_precision = 42;
        let proto = to_hll_proto(&state);
        assert!(proto.has_sparse_precision_or_num_buckets());
        assert_eq!(proto.sparse_precision_or_num_buckets(), 42);
    }

    #[test]
    fn test_serialize_data() {
        let mut state = State::default();

        let proto = to_hll_proto(&state);
        assert!(!proto.has_data());

        state.data = Some(vec![1, 2, 3]);
        let r = state.to_byte_array().unwrap();
        let p = AggregatorStateProto::parse_from_bytes(&r).unwrap();
        let u = p.unknown_fields().get(112);
        assert!(u.is_some());
        let field = u.unwrap();
        match field {
            UnknownValueRef::LengthDelimited(data) => {
                let h = HyperLogLogPlusUniqueStateProto::parse_from_bytes(data).unwrap();
                assert!(h.has_data());
            }
            _ => panic!("Wrong field type"),
        };
        let proto = to_hll_proto(&state);
        assert!(proto.has_data());
        assert_eq!(proto.data(), vec![1, 2, 3]);
    }

    #[test]
    fn test_serialize_sparse_data() {
        let mut state = State::default();

        let proto = to_hll_proto(&state);
        assert!(!proto.has_sparse_data());

        state.sparse_data = Some(vec![1, 2, 3]);
        let proto = to_hll_proto(&state);
        assert!(proto.has_sparse_data());
        assert_eq!(proto.sparse_data(), vec![1, 2, 3]);
    }
}
