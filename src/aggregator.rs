use crate::{error::SketchError, protos::AggregatorStateProto};

pub trait Aggregator<R, A: Aggregator<R, A>> {
    fn merge_aggregator(&mut self, other: A) -> Result<(), SketchError>;

    fn merge_proto(&mut self, proto: AggregatorStateProto) -> Result<(), SketchError>;

    fn merge_bytes(&mut self, data: &[u8]) -> Result<(), SketchError>;

    fn num_values(&self) -> u64;

    fn result(&self) -> Result<R, SketchError>;

    fn serialize_to_bytes(self) -> Result<Vec<u8>, SketchError>;

    fn serialize_to_proto(self) -> Result<AggregatorStateProto, SketchError>;
}
