#![allow(dead_code)]

// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

use std::rc::Rc;

use super::Error;

use base64::prelude::*;
use j4rs::{errors::J4RsError, Instance, InvocationArg, Jvm};

pub struct HyperLogLogPlusPlusBuilder {
    jvm: Rc<Jvm>,
    builder: Instance,
}

impl HyperLogLogPlusPlusBuilder {
    pub(crate) fn for_jvm(jvm: Rc<Jvm>) -> Result<Self, Error> {
        let builder = jvm.create_instance(
            "com.google.zetasketch.HyperLogLogPlusPlus$Builder",
            InvocationArg::empty(),
        )?;

        Ok(Self { jvm, builder })
    }

    pub fn normal_precision(self, precision: i32) -> Result<Self, Error> {
        self.jvm.invoke(
            &self.builder,
            "normalPrecision",
            &[InvocationArg::try_from(precision)?],
        )?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn sparse_precision(self, precision: i32) -> Result<Self, Error> {
        self.jvm.invoke(
            &self.builder,
            "sparsePrecision",
            &[InvocationArg::try_from(precision)?],
        )?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn no_sparse_mode(self) -> Result<Self, Error> {
        self.jvm
            .invoke(&self.builder, "noSparseMode", InvocationArg::empty())?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn build_for_bytes(self) -> Result<HyperLogLogPlusPlus<Vec<u8>>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForBytes", InvocationArg::empty())?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_integers(self) -> Result<HyperLogLogPlusPlus<u32>, Error> {
        let aggregator =
            self.jvm
                .invoke(&self.builder, "buildForIntegers", InvocationArg::empty())?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_longs(self) -> Result<HyperLogLogPlusPlus<u64>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForLongs", InvocationArg::empty())?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_strings(self) -> Result<HyperLogLogPlusPlus<String>, Error> {
        let aggregator =
            self.jvm
                .invoke(&self.builder, "buildForStrings", InvocationArg::empty())?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }
}

pub struct HyperLogLogPlusPlus<T> {
    jvm: Rc<Jvm>,
    hll: Instance,
    _marker: std::marker::PhantomData<T>,
}

impl<T> HyperLogLogPlusPlus<T> {
    pub(crate) fn new(jvm: Rc<Jvm>, hll: Instance) -> Self {
        Self {
            jvm,
            hll,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn for_proto(jvm: Rc<Jvm>, bytes: &[u8]) -> Result<Self, Error> {
        let bytes_arg = jvm.create_java_array(
            "byte",
            bytes
                .iter()
                .map(|b| InvocationArg::try_from(*b as i8)?.into_primitive())
                .collect::<Result<Vec<_>, J4RsError>>()?
                .as_slice(),
        )?;

        let hll = jvm.invoke_static(
            "com.google.zetasketch.HyperLogLogPlusPlus",
            "forProto",
            &[InvocationArg::from(bytes_arg)],
        )?;

        Ok(Self {
            jvm,
            hll,
            _marker: std::marker::PhantomData,
        })
    }

    fn add_impl<U>(&self, value: U) -> Result<(), Error>
    where
        InvocationArg: TryFrom<U>,
    {
        self.jvm.invoke(
            &self.hll,
            "add",
            &[InvocationArg::try_from(value).map_err(|_| {
                Error::JavaError(j4rs::errors::J4RsError::ParseError(
                    "Failed to convert value to InvocationArg".to_string(),
                ))
            })?],
        )?;

        Ok(())
    }

    pub fn merge(&self, other: Self) -> Result<(), Error> {
        self.jvm
            .invoke(&self.hll, "merge", &[InvocationArg::from(other.hll)])?;

        Ok(())
    }

    pub fn merge_proto_bytes(&self, proto_bytes: &[u8]) -> Result<(), Error> {
        let jarray = self.jvm.create_java_array(
            "byte",
            proto_bytes
                .iter()
                .map(|b| InvocationArg::try_from(*b as i8)?.into_primitive())
                .collect::<Result<Vec<_>, J4RsError>>()?
                .as_slice(),
        )?;

        self.jvm
            .invoke(&self.hll, "merge", &[InvocationArg::from(jarray)])?;

        Ok(())
    }

    pub fn result(&self) -> Result<i64, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "longResult", InvocationArg::empty())?;
        Ok(self.jvm.to_rust(result)?)
    }

    pub fn num_values(&self) -> Result<u64, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "numValues", InvocationArg::empty())?;
        Ok(self.jvm.to_rust(result)?)
    }

    pub fn get_normal_precision(&self) -> Result<i32, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "getNormalPrecision", InvocationArg::empty())?;
        self.jvm.to_rust(result).map_err(Error::JavaError)
    }

    pub fn get_sparse_precision(&self) -> Result<i32, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "getSparsePrecision", InvocationArg::empty())?;
        self.jvm.to_rust(result).map_err(Error::JavaError)
    }

    pub fn serialize_to_byte_array(&self) -> Result<Vec<u8>, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "serializeToByteArray", InvocationArg::empty())?;
        let enc_str: String = self.jvm.to_rust(result)?;
        Ok(BASE64_STANDARD.decode(enc_str).unwrap())
    }
}

impl HyperLogLogPlusPlus<String> {
    pub fn add(&self, value: String) -> Result<(), Error> {
        self.add_impl(value)
    }

    pub fn add_str(&self, value: &str) -> Result<(), Error> {
        self.add_impl(value)
    }
}

impl HyperLogLogPlusPlus<Vec<u8>> {
    pub fn add(&self, value: Vec<u8>) -> Result<(), Error> {
        let bytes = self.jvm.create_java_array(
            "byte",
            value
                .iter()
                .map(|b| InvocationArg::try_from(*b as i8)?.into_primitive())
                .collect::<Result<Vec<_>, J4RsError>>()?
                .as_slice(),
        )?;

        self.add_impl(bytes)
    }
}

impl HyperLogLogPlusPlus<u64> {
    pub fn add(&self, value: u64) -> Result<(), Error> {
        self.add_impl(value as i64)
    }
}

impl HyperLogLogPlusPlus<u32> {
    pub fn add(&self, value: u32) -> Result<(), Error> {
        self.add_impl(value as i32)
    }
}

#[cfg(test)]
mod tests {
    use crate::Zetasketch;

    #[test]
    fn test_add_str() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let builder = zetasketch.builder().expect("Failed to create builder");
        let hll = builder.build_for_strings().expect("Failed to build HLL");
        hll.add_str("a").expect("Failed to add str");
        hll.add_str("b").expect("Failed to add str");
        assert!(hll.result().expect("Failed to get result") == 2);

        hll.add_str("a").expect("Failed to add str");
        assert!(hll.result().expect("Failed to get result") == 2);

        hll.add_str("c").expect("Failed to add str");
        assert!(hll.result().expect("Failed to get result") == 3);
    }

    #[test]
    fn test_add_i64() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let builder = zetasketch.builder().expect("Failed to create builder");
        let hll = builder.build_for_longs().expect("Failed to build HLL");
        hll.add(1).expect("Failed to add i64");
        assert!(hll.result().expect("Failed to get result") == 1);

        hll.add(2).expect("Failed to add i64");
        assert!(hll.result().expect("Failed to get result") == 2);

        hll.add(2).expect("Failed to add i64");
        assert!(hll.result().expect("Failed to get result") == 2);
    }

    #[test]
    fn test_add_i32() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let builder = zetasketch.builder().expect("Failed to create builder");
        let hll = builder.build_for_integers().expect("Failed to build HLL");
        hll.add(1).expect("Failed to add i32");
        assert!(hll.result().expect("Failed to get result") == 1);

        hll.add(2).expect("Failed to add i32");
        assert!(hll.result().expect("Failed to get result") == 2);

        hll.add(2).expect("Failed to add i32");
        assert!(hll.result().expect("Failed to get result") == 2);
    }

    #[test]
    fn test_add_bytes() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let builder = zetasketch.builder().expect("Failed to create builder");
        let hll = builder.build_for_bytes().expect("Failed to build HLL");
        hll.add(vec![1, 2, 3]).expect("Failed to add bytes");
        assert!(hll.result().expect("Failed to get result") == 1);

        hll.add(vec![1, 2, 3]).expect("Failed to add bytes");
        assert!(hll.result().expect("Failed to get result") == 1);

        hll.add(vec![4, 5, 6]).expect("Failed to add bytes");
        assert!(hll.result().expect("Failed to get result") == 2);
    }

    #[test]
    fn test_merge() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let hll1 = zetasketch
            .builder()
            .expect("Failed to create builder")
            .build_for_strings()
            .expect("Failed to build HLL");
        let hll2 = zetasketch
            .builder()
            .expect("Failed to create builder")
            .build_for_strings()
            .expect("Failed to build HLL");

        hll1.add_str("a").expect("Failed to add str");
        hll1.add_str("b").expect("Failed to add str");
        assert!(hll1.result().expect("Failed to get result") == 2);

        hll2.add_str("b").expect("Failed to add str");
        hll2.add_str("c").expect("Failed to add str");
        assert!(hll2.result().expect("Failed to get result") == 2);

        hll1.merge(hll2).expect("Failed to merge");
        assert!(hll1.result().expect("Failed to get result") == 3);
    }

    #[test]
    fn test_serialize_to_proto() {
        let zetasketch = Zetasketch::new().expect("Failed to create Zetasketch JVM");
        let builder = zetasketch.builder().expect("Failed to create builder");
        let hll = builder.build_for_strings().expect("Failed to build HLL");
        hll.add_str("a").expect("Failed to add str");

        let proto = hll
            .serialize_to_byte_array()
            .expect("Failed to serialize to proto");

        let hll2 = zetasketch
            .builder()
            .expect("Failed to create builder")
            .build_for_strings()
            .expect("Failed to build HLL");
        hll2.merge_proto_bytes(&proto)
            .expect("Failed to merge proto");

        assert!(hll2.result().expect("Failed to get result") == 1);
        hll2.add_str("a").expect("Failed to add str");
        assert!(hll2.result().expect("Failed to get result") == 1);
    }
}
