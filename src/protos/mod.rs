// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

// This file will be populated by prost-build in build.rs

// Include the generated Protobuf code.
// The actual filenames will be based on your .proto package structure.
// For example, if your aggregator.proto has `package google.protos.zetasketch;`
// the generated module will likely be `google.protos.zetasketch`.

// Assuming the proto files `aggregator.proto` and `hllplus_unique.proto`
// declare a package like `com.google.protos.zetasketch` or similar.
// The include path will depend on the package name in the .proto files.
// If they are in `google.protos.zetasketch`:
// pub mod google {
//     pub mod protos {
//         pub mod zetasketch {
//             include!(concat!(env!("OUT_DIR"), "/google.protos.zetasketch.rs"));
//         }
//     }
// }
// For now, let's assume a simpler structure or that the user will adjust this.
// If the .proto files define `package zetasketch;`, it would be:
// pub mod zetasketch {
//    include!(concat!(env!("OUT_DIR"), "/zetasketch.rs"));
// }

// Placeholder for the actual generated code include.
// The user will need to ensure proto files are copied and build.rs runs successfully.
// The generated file will be in OUT_DIR, and we're telling build.rs to put it in `src/protos`.
// So, we'll reference it from there.

// This assumes that aggregator.proto and hllplus_unique.proto might generate
// separate .rs files or a combined one based on their package definitions.
// Let's assume a common package `zetasketch_protos` for simplicity of the generated file name.
// The `build.rs` is configured with `out_dir("src/protos")`.

// Assuming the .proto files use `package com.google.protos.zetasketch;`
// This will generate a single `com.google.protos.zetasketch.rs` file.
#[allow(clippy::all)] // To silence clippy warnings from generated code
pub mod zetasketch {
    include!(concat!(env!("OUT_DIR"), "/protos/mod.rs"));
}

pub use zetasketch::aggregator::default_ops_type::Id as DefaultOpsTypeId;
pub use zetasketch::aggregator::AggregatorStateProto;
pub use zetasketch::aggregator::AggregatorStatsProto;
pub use zetasketch::aggregator::AggregatorType;
pub use zetasketch::aggregator::AggregatorValueStatsProto;
pub use zetasketch::aggregator::DefaultOpsType;

pub use zetasketch::custom_value_type::custom_value_type::Id as CustomValueTypeId;
pub use zetasketch::custom_value_type::CustomValueType;

pub use zetasketch::hllplus_unique::HyperLogLogPlusUniqueStateProto;
pub mod exts {
    pub use super::zetasketch::hllplus_unique::exts::hyperloglogplus_unique_state;
}
