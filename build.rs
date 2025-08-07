// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

use std::env;
use std::path::PathBuf;

fn build_protobufs() -> Result<(), Box<dyn std::error::Error>> {
    let root_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let proto_dir = root_dir.join("src/protos");

    protobuf_codegen::Codegen::new()
        .protoc()
        .include(&proto_dir)
        .input(proto_dir.join("aggregator.proto"))
        .input(proto_dir.join("annotation.proto"))
        .input(proto_dir.join("custom-value-type.proto"))
        .input(proto_dir.join("hllplus-unique.proto"))
        .input(proto_dir.join("unique-stats.proto"))
        .cargo_out_dir("protos")
        .run()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    build_protobufs()?;

    Ok(())
}
