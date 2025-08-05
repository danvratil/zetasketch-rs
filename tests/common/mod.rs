use j4rs::{JavaOpt, Jvm, JvmBuilder};
use std::sync::Arc;
use thiserror::Error;

mod hyperloglog;

pub use hyperloglog::{HyperLogLogPlusPlusBuilder, HyperLogLogPlusPlus};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Java error: {0}")]
    JavaError(#[from] j4rs::errors::J4RsError),
    #[error("Proto error: {0}")]
    ProtoError(#[from] protobuf::Error),
}

pub struct Zetasketch {
    jvm: Arc<Jvm>,
}

impl Zetasketch {
    pub fn new() -> Result<Self, Error> {
        let jvm = JvmBuilder::new()
            .java_opt(JavaOpt::new("--illegal-access=warn"))
            .build()?;

        Ok(Self { jvm: Arc::new(jvm) })
    }

    pub fn builder(&self) -> Result<HyperLogLogPlusPlusBuilder, Error> {
        HyperLogLogPlusPlusBuilder::for_jvm(Arc::clone(&self.jvm))
    }

    pub fn hll_for_bytes<T>(&self, bytes: &[u8]) -> Result<HyperLogLogPlusPlus<T>, Error> {
        HyperLogLogPlusPlus::for_proto(Arc::clone(&self.jvm), bytes)
    }
}
