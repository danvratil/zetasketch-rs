use j4rs::{JavaOpt, Jvm, JvmBuilder};
use std::sync::Arc;
use thiserror::Error;

mod hyperloglog;

pub use hyperloglog::HyperLogLogPlusPlusBuilder;

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("Java error: {0}")]
    JavaError(j4rs::errors::J4RsError),
    #[error("Proto error: {0}")]
    ProtoError(prost::DecodeError),
}

pub struct Zetasketch {
    jvm: Arc<Jvm>,
}

impl Zetasketch {
    pub fn new() -> Result<Self, Error> {
        let jvm = JvmBuilder::new()
            .java_opt(JavaOpt::new("--illegal-access=warn"))
            .build()
            .map_err(Error::JavaError)?;

        Ok(Self { jvm: Arc::new(jvm) })
    }

    pub fn builder(&self) -> Result<HyperLogLogPlusPlusBuilder, Error> {
        HyperLogLogPlusPlusBuilder::for_jvm(Arc::clone(&self.jvm))
    }
}

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/zetasketch.rs"));
}