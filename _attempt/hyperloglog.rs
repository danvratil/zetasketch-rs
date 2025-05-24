use std::sync::Arc;

use crate::{proto, Error};

use base64::prelude::*;
use j4rs::{errors::J4RsError, Instance, InvocationArg, Jvm};
use prost::Message;

pub struct HyperLogLogPlusPlusBuilder {
    jvm: Arc<Jvm>,
    builder: Instance,
}

impl HyperLogLogPlusPlusBuilder {
    pub(crate) fn for_jvm(jvm: Arc<Jvm>) -> Result<Self, Error> {
        let builder = jvm
            .create_instance(
                "com.google.zetasketch.HyperLogLogPlusPlus$Builder",
                InvocationArg::empty(),
            )
            .map_err(Error::JavaError)?;

        Ok(Self { jvm, builder })
    }

    pub fn normal_precision(self, precision: i32) -> Result<Self, Error> {
        self.jvm
            .invoke(
                &self.builder,
                "normalPrecision",
                &[InvocationArg::try_from(precision).map_err(Error::JavaError)?],
            )
            .map_err(Error::JavaError)?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn sparse_precision(self, precision: i32) -> Result<Self, Error> {
        self.jvm
            .invoke(
                &self.builder,
                "sparsePrecision",
                &[InvocationArg::try_from(precision).map_err(Error::JavaError)?],
            )
            .map_err(Error::JavaError)?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn no_sparse_mode(self) -> Result<Self, Error> {
        self.jvm
            .invoke(&self.builder, "noSparseMode", InvocationArg::empty())
            .map_err(Error::JavaError)?;

        Ok(Self {
            jvm: self.jvm,
            builder: self.builder,
        })
    }

    pub fn build_for_bytes(self) -> Result<HyperLogLogPlusPlus<Vec<u8>>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForBytes", InvocationArg::empty())
            .map_err(Error::JavaError)?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_integers(self) -> Result<HyperLogLogPlusPlus<u32>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForIntegers", InvocationArg::empty())
            .map_err(Error::JavaError)?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_longs(self) -> Result<HyperLogLogPlusPlus<u64>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForLongs", InvocationArg::empty())
            .map_err(Error::JavaError)?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }

    pub fn build_for_strings(self) -> Result<HyperLogLogPlusPlus<String>, Error> {
        let aggregator = self
            .jvm
            .invoke(&self.builder, "buildForStrings", InvocationArg::empty())
            .map_err(Error::JavaError)?;

        Ok(HyperLogLogPlusPlus::new(self.jvm, aggregator))
    }
}

pub struct HyperLogLogPlusPlus<T> {
    jvm: Arc<Jvm>,
    hll: Instance,
    _marker: std::marker::PhantomData<T>,
}

impl<T> HyperLogLogPlusPlus<T> {
    pub(crate) fn new(jvm: Arc<Jvm>, hll: Instance) -> Self {
        Self {
            jvm,
            hll,
            _marker: std::marker::PhantomData,
        }
    }

    fn add_impl<U>(&self, value: U) -> Result<(), Error>
    where
        InvocationArg: TryFrom<U>,
    {
        self.jvm
            .invoke(
                &self.hll,
                "add",
                &[InvocationArg::try_from(value).map_err(|_| {
                    Error::JavaError(j4rs::errors::J4RsError::ParseError(
                        "Failed to convert value to InvocationArg".to_string(),
                    ))
                })?],
            )
            .map_err(Error::JavaError)?;

        Ok(())
    }

    pub fn merge(&self, other: Self) -> Result<(), Error> {
        self.jvm
            .invoke(
                &self.hll,
                "merge",
                &[InvocationArg::try_from(other.hll).unwrap()],
            )
            .map_err(Error::JavaError)?;

        Ok(())
    }

    pub fn merge_proto(&self, state: proto::AggregatorStateProto) -> Result<(), Error> {
        self.merge_proto_bytes(&state.encode_to_vec())
    }

    pub fn merge_proto_bytes(&self, proto_bytes: &[u8]) -> Result<(), Error> {
        let jarray = self
            .jvm
            .create_java_array(
                "byte",
                proto_bytes
                    .iter()
                    .map(|b| {
                        Ok::<_, J4RsError>(InvocationArg::try_from(*b as i8)?.into_primitive()?)
                    })
                    .collect::<Result<Vec<_>, J4RsError>>()
                    .map_err(Error::JavaError)?
                    .as_slice(),
            )
            .map_err(Error::JavaError)?;

        let arg = InvocationArg::from(jarray);

        self.jvm
            .invoke(&self.hll, "merge", &[&arg])
            .map_err(Error::JavaError)?;

        Ok(())
    }

    pub fn result(&self) -> Result<u64, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "longResult", InvocationArg::empty())
            .map_err(Error::JavaError)?;
        self.jvm.to_rust(result).map_err(Error::JavaError)
    }

    pub fn num_values(&self) -> Result<u64, Error> {
        self.result()
    }

    pub fn get_normal_precision(&self) -> Result<i32, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "getNormalPrecision", InvocationArg::empty())
            .map_err(Error::JavaError)?;
        self.jvm.to_rust(result).map_err(Error::JavaError)
    }

    pub fn get_sparse_precision(&self) -> Result<i32, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "getSparsePrecision", InvocationArg::empty())
            .map_err(Error::JavaError)?;
        self.jvm.to_rust(result).map_err(Error::JavaError)
    }

    pub fn serialize_to_proto(&self) -> Result<proto::AggregatorStateProto, Error> {
        let bytes = self.serialize_to_byte_array()?;
        proto::AggregatorStateProto::decode(&mut bytes.as_slice()).map_err(Error::ProtoError)
    }

    pub fn serialize_to_byte_array(&self) -> Result<Vec<u8>, Error> {
        let result = self
            .jvm
            .invoke(&self.hll, "serializeToByteArray", InvocationArg::empty())
            .map_err(Error::JavaError)?;
        let enc_str: String = self.jvm.to_rust(result).map_err(Error::JavaError)?;
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
        let bytes = self
            .jvm
            .create_java_array(
                "byte",
                value
                    .iter()
                    .map(|b| {
                        Ok::<_, J4RsError>(InvocationArg::try_from(*b as i8)?.into_primitive()?)
                    })
                    .collect::<Result<Vec<_>, J4RsError>>()
                    .map_err(Error::JavaError)?
                    .as_slice(),
            )
            .map_err(Error::JavaError)?;

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
    use assert2::assert;

    #[test]
    fn test_add_str() {
        let zetasketch = Zetasketch::new().unwrap();
        let builder = zetasketch.builder().unwrap();
        let hll = builder.build_for_strings().unwrap();
        hll.add_str("a").unwrap();
        hll.add_str("b").unwrap();
        assert!(hll.result().unwrap() == 2);

        hll.add_str("a").unwrap();
        assert!(hll.result().unwrap() == 2);

        hll.add_str("c").unwrap();
        assert!(hll.result().unwrap() == 3);
    }

    #[test]
    fn test_add_i64() {
        let zetasketch = Zetasketch::new().unwrap();
        let builder = zetasketch.builder().unwrap();
        let hll = builder.build_for_longs().unwrap();
        hll.add(1).unwrap();
        assert!(hll.result().unwrap() == 1);

        hll.add(2).unwrap();
        assert!(hll.result().unwrap() == 2);

        hll.add(2).unwrap();
        assert!(hll.result().unwrap() == 2);
    }

    #[test]
    fn test_add_i32() {
        let zetasketch = Zetasketch::new().unwrap();
        let builder = zetasketch.builder().unwrap();
        let hll = builder.build_for_integers().unwrap();
        hll.add(1).unwrap();
        assert!(hll.result().unwrap() == 1);

        hll.add(2).unwrap();
        assert!(hll.result().unwrap() == 2);

        hll.add(2).unwrap();
        assert!(hll.result().unwrap() == 2);
    }

    #[test]
    fn test_add_bytes() {
        let zetasketch = Zetasketch::new().unwrap();
        let builder = zetasketch.builder().unwrap();
        let hll = builder.build_for_bytes().unwrap();
        hll.add(vec![1, 2, 3]).unwrap();
        assert!(hll.result().unwrap() == 1);

        hll.add(vec![1, 2, 3]).unwrap();
        assert!(hll.result().unwrap() == 1);

        hll.add(vec![4, 5, 6]).unwrap();
        assert!(hll.result().unwrap() == 2);
    }

    #[test]
    fn test_merge() {
        let zetasketch = Zetasketch::new().unwrap();
        let hll1 = zetasketch.builder().unwrap().build_for_strings().unwrap();
        let hll2 = zetasketch.builder().unwrap().build_for_strings().unwrap();

        hll1.add_str("a").unwrap();
        hll1.add_str("b").unwrap();
        assert!(hll1.result().unwrap() == 2);

        hll2.add_str("b").unwrap();
        hll2.add_str("c").unwrap();
        assert!(hll2.result().unwrap() == 2);

        hll1.merge(hll2).unwrap();
        assert!(hll1.result().unwrap() == 3);
    }

    #[test]
    fn test_serialize_to_proto() {
        let zetasketch = Zetasketch::new().unwrap();
        let builder = zetasketch.builder().unwrap();
        let hll = builder.build_for_strings().unwrap();
        hll.add_str("a").unwrap();

        let proto = hll.serialize_to_proto().unwrap();

        let hll2 = zetasketch.builder().unwrap().build_for_strings().unwrap();
        hll2.merge_proto(proto).unwrap();

        assert!(hll2.result().unwrap() == 1);
        hll2.add_str("a").unwrap();
        assert!(hll2.result().unwrap() == 1);
    }
}
