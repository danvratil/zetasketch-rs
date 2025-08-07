// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

use j4rs::{JavaOpt, Jvm, JvmBuilder};
use std::rc::Rc;
use thiserror::Error;

mod hyperloglog;

pub use hyperloglog::{HyperLogLogPlusPlus, HyperLogLogPlusPlusBuilder};

#[derive(Debug, Error)]
pub enum Error {
    #[error("Java error: {0}")]
    JavaError(#[from] j4rs::errors::J4RsError),
    #[error("Proto error: {0}")]
    ProtoError(#[from] protobuf::Error),
}

pub struct Zetasketch {
    jvm: Rc<Jvm>,
}

impl Zetasketch {
    pub fn new() -> Result<Self, Error> {
        let jvm = JvmBuilder::new()
            .java_opt(JavaOpt::new("--illegal-access=warn"))
            .build()?;

        let rc_jvm = Rc::new(jvm);
        Ok(Self { jvm: rc_jvm })
    }

    pub fn builder(&self) -> Result<HyperLogLogPlusPlusBuilder, Error> {
        HyperLogLogPlusPlusBuilder::for_jvm(Rc::clone(&self.jvm))
    }

    pub fn hll_for_bytes<T>(&self, bytes: &[u8]) -> Result<HyperLogLogPlusPlus<T>, Error> {
        HyperLogLogPlusPlus::for_proto(Rc::clone(&self.jvm), bytes)
    }
}
