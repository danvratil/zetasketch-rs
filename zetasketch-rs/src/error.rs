use thiserror::Error;

#[derive(Error, Debug)]
pub enum SketchError {
    #[error("Incompatible precision: {0}")]
    IncompatiblePrecision(String),
    #[error("Invalid HLL++ state: {0}")]
    InvalidState(String),
    #[error("Proto serialization error: {0}")]
    ProtoSerialization(protobuf::Error),
    #[error("Proto deserialization error: {0}")]
    ProtoDeserialization(protobuf::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid argument: {0}")]
    IllegalArgument(String),
    // TODO: Add more specific error types as needed.
}
