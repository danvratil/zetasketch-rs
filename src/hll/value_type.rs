// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

use crate::protos::{CustomValueTypeId, DefaultOpsTypeId};
use protobuf::Enum;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    DefaultOpsType(DefaultOpsTypeId),
    CustomType(i32),
    Unknown,
}

impl From<i32> for ValueType {
    fn from(value: i32) -> Self {
        if value == 0 {
            ValueType::Unknown
        } else if value < 1000 {
            match DefaultOpsTypeId::from_i32(value) {
                Some(id) => ValueType::DefaultOpsType(id),
                None => ValueType::Unknown,
            }
        } else {
            ValueType::CustomType(value)
        }
    }
}

impl From<CustomValueTypeId> for ValueType {
    fn from(value: CustomValueTypeId) -> Self {
        #[allow(unreachable_patterns)]
        match value {
            CustomValueTypeId::UNKNOWN => ValueType::Unknown,
            _ => ValueType::CustomType(value as i32),
        }
    }
}

impl From<DefaultOpsTypeId> for ValueType {
    fn from(value: DefaultOpsTypeId) -> Self {
        match value {
            DefaultOpsTypeId::UNKNOWN => ValueType::Unknown,
            _ => ValueType::DefaultOpsType(value),
        }
    }
}

impl From<ValueType> for i32 {
    fn from(value: ValueType) -> i32 {
        match value {
            ValueType::Unknown => 0,
            ValueType::DefaultOpsType(value) => value as i32,
            ValueType::CustomType(value) => value,
        }
    }
}

impl From<ValueType> for DefaultOpsTypeId {
    fn from(val: ValueType) -> Self {
        match val {
            ValueType::Unknown => DefaultOpsTypeId::UNKNOWN,
            ValueType::DefaultOpsType(id) => id,
            ValueType::CustomType(_) => panic!("Cannot convert custom type to DefaultOpsTypeId"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_unknown() {
        assert_eq!(i32::from(ValueType::Unknown), 0);
        assert_eq!(
            Into::<i32>::into(ValueType::Unknown),
            DefaultOpsTypeId::UNKNOWN.value()
        );
        assert_eq!(ValueType::from(0), ValueType::Unknown);
    }

    #[test]
    pub fn test_from_default_ops_type_number() {
        let value_type = ValueType::from(DefaultOpsTypeId::INT32);
        assert!(matches!(
            value_type,
            ValueType::DefaultOpsType(DefaultOpsTypeId::INT32)
        ));
        assert_eq!(i32::from(value_type), DefaultOpsTypeId::INT32.value());
    }

    #[test]
    pub fn test_from_other_number() {
        let value_type = ValueType::from(12345);
        assert_eq!(i32::from(value_type), 12345);
        assert!(matches!(value_type, ValueType::CustomType(12345)));
    }
}
