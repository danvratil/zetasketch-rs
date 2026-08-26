// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

use j4rs::{JavaOpt, Jvm, JvmBuilder};
use std::rc::Rc;
use thiserror::Error;

mod hyperloglog;
mod jassets;

pub use hyperloglog::{HyperLogLogPlusPlus, HyperLogLogPlusPlusBuilder};
pub use jassets::{deploy_maven_artifacts, ensure_j4rs_base_path};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Java error: {0}")]
    JavaError(#[from] j4rs::errors::J4RsError),
    #[error("Proto error: {0}")]
    ProtoError(#[from] protobuf::Error),
    #[error("{0}")]
    Setup(String),
}

pub struct Zetasketch {
    jvm: Rc<Jvm>,
}

impl Zetasketch {
    pub fn new() -> Result<Self, Error> {
        let base = ensure_j4rs_base_path()?;
        let jvm = JvmBuilder::new()
            .with_base_path(&base)
            .java_opt(JavaOpt::new("-XX:+IgnoreUnrecognizedVMOptions"))
            .java_opt(JavaOpt::new("--illegal-access=warn"))
            .java_opt(JavaOpt::new("--enable-native-access=ALL-UNNAMED"))
            .build()?;
        deploy_maven_artifacts(&jvm)?;

        Ok(Self { jvm: Rc::new(jvm) })
    }

    pub fn builder(&self) -> Result<HyperLogLogPlusPlusBuilder, Error> {
        HyperLogLogPlusPlusBuilder::for_jvm(Rc::clone(&self.jvm))
    }

    pub fn hll_for_bytes<T>(&self, bytes: &[u8]) -> Result<HyperLogLogPlusPlus<T>, Error> {
        HyperLogLogPlusPlus::for_proto(Rc::clone(&self.jvm), bytes)
    }
}
