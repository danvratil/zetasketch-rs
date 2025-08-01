use std::env;
use std::path::PathBuf;
use j4rs::{JvmBuilder, MavenArtifact};

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

fn download_java_artifacts() -> Result<(), Box<dyn std::error::Error>> {
    let jvm = JvmBuilder::new().build()?;
    jvm.deploy_artifact(&MavenArtifact::from("com.google.zetasketch:zetasketch:0.1.0"))?;
    jvm.deploy_artifact(&MavenArtifact::from("it.unimi.dsi:fastutil:8.2.2"))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    build_protobufs()?;
    download_java_artifacts()?;

    Ok(())
}
