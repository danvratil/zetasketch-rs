use anyhow::{Context, Result};
use fs_extra::dir::{copy, CopyOptions};
use j4rs::{JvmBuilder, MavenArtifact};
use prost_build;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

fn replace_in_file(path: &Path, pattern: &str, replacement: &str) -> std::io::Result<()> {
    let content = fs::read_to_string(path)?;
    let re = Regex::new(pattern).expect("Invalid regex pattern");
    let new_content = re.replace_all(&content, replacement);
    fs::write(path, new_content.as_bytes())?;
    Ok(())
}

fn copy_to_build_dir() -> Result<PathBuf> {
    let source_dir = Path::new("google-zetasketch");
    let build_dir = PathBuf::from(&std::env::var("OUT_DIR")?);

    copy(
        source_dir,
        &build_dir,
        &CopyOptions {
            overwrite: true,
            skip_exist: false,
            copy_inside: true,
            ..Default::default()
        },
    )
    .context("Failed to copy google-zetasketch directory")?;

    Ok(build_dir.join("google-zetasketch"))
}

fn patch_protobuf(build_dir: &Path) -> Result<()> {
    let proto_file = build_dir.join("proto/aggregator.proto");
    replace_in_file(&proto_file, r"reserved 0,", "reserved")?;

    Ok(())
}

fn deploy_artifacts() -> Result<()> {
    // Now create a j4rs JVM to deploy the dependencies
    // Ideally we would parse the deps from the build.gradle file, but this is easier for now...
    const DEPENDENCIES: [&str; 8] = [
        "com.google.zetasketch:zetasketch:0.1.0",
        "com.google.auto.value:auto-value-annotations:1.6.3",
        "com.google.code.findbugs:jsr305:3.0.2",
        "com.google.errorprone:error_prone_annotations:2.3.2",
        "com.google.guava:guava:28.0-jre",
        "com.google.protobuf:protobuf-java:3.6.0",
        "it.unimi.dsi:fastutil:8.2.2",
        "org.checkerframework:checker-qual:2.8.1",
    ];
    let mut builder = JvmBuilder::new();
    let jvm = builder.build().context("Failed to build JVM")?;
    for dependency in DEPENDENCIES {
        jvm.deploy_artifact(&MavenArtifact::from(dependency))
            .context("Failed to deploy artifact")?;
    }

    Ok(())
}

fn compile_protobuf(build_dir: &Path) -> Result<()> {
    prost_build::Config::new()
        .compile_protos(
            &[
                build_dir.join("proto/aggregator.proto"),
                build_dir.join("proto/annotation.proto"),
                build_dir.join("proto/custom-value-type.proto"),
                build_dir.join("proto/hllplus-unique.proto"),
                build_dir.join("proto/unique-stats.proto"),
            ],
            &[build_dir.join("proto")],
        )
        .context("Failed to compile protobuf")?;

    Ok(())
}

fn main() {
    let build_dir = copy_to_build_dir().expect("Failed to copy to build dir");
    patch_protobuf(&build_dir).expect("Failed to patch protobuf");

    deploy_artifacts().expect("Failed to install Maven artifacts");
    compile_protobuf(&build_dir).expect("Failed to compile protobuf");
}
