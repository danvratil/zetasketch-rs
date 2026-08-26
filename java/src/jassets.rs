// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT

//! Bootstrap j4rs's `jassets` directory and deploy the ZetaSketch Maven jars.
//!
//! j4rs 0.25 refuses to start a JVM unless it can see a `jassets` directory
//! containing its fat jar. That directory is normally created by *j4rs's own*
//! `build.rs`, which does not re-run when the crate is already compiled. A
//! clean `target/` (or a deleted `jassets/`) therefore used to fail in our
//! `build.rs` with `Can not find jassets directory` before any Maven fetch.
//!
//! We create `jassets` ourselves next to the test/bench binary, copy the
//! bundled j4rs jar out of the cargo registry if needed, and download the
//! ZetaSketch artifacts when the JVM is first created.

use std::fs;
use std::path::{Path, PathBuf};

use j4rs::{Jvm, MavenArtifact};

use crate::Error;

/// Directory that should contain a `jassets/` subdirectory (typically
/// `target/debug` or `target/release`).
pub fn ensure_j4rs_base_path() -> Result<PathBuf, Error> {
    let base = infer_profile_dir()?;
    let jassets = base.join("jassets");
    fs::create_dir_all(&jassets)
        .map_err(|e| Error::Setup(format!("failed to create {}: {e}", jassets.display())))?;

    if !jassets_has_j4rs_jar(&jassets) {
        let src = find_bundled_j4rs_jar().ok_or_else(|| {
            Error::Setup(
                "could not locate j4rs-*-jar-with-dependencies.jar in the cargo \
                 registry; is j4rs downloaded?"
                    .to_string(),
            )
        })?;
        let dest = jassets.join(src.file_name().unwrap());
        fs::copy(&src, &dest).map_err(|e| {
            Error::Setup(format!(
                "failed to copy {} to {}: {e}",
                src.display(),
                dest.display()
            ))
        })?;
    }

    Ok(base)
}

/// Fetch `zetasketch` and `fastutil` into `jassets` and add them to the live JVM.
pub fn deploy_maven_artifacts(jvm: &Jvm) -> Result<(), Error> {
    jvm.deploy_artifact(&MavenArtifact::from(
        "com.google.zetasketch:zetasketch:0.1.0",
    ))?;
    jvm.deploy_artifact(&MavenArtifact::from("it.unimi.dsi:fastutil:8.2.2"))?;
    Ok(())
}

fn infer_profile_dir() -> Result<PathBuf, Error> {
    let exe =
        std::env::current_exe().map_err(|e| Error::Setup(format!("current_exe failed: {e}")))?;
    let mut dir = exe
        .parent()
        .ok_or_else(|| Error::Setup(format!("{} has no parent", exe.display())))?
        .to_path_buf();
    // `cargo test` binaries live in `target/<profile>/deps/`.
    if dir
        .file_name()
        .is_some_and(|n| n == "deps" || n == "examples")
    {
        dir.pop();
    }
    Ok(dir)
}

fn jassets_has_j4rs_jar(jassets: &Path) -> bool {
    let Ok(entries) = fs::read_dir(jassets) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with("j4rs-") && name.ends_with("-jar-with-dependencies.jar")
    })
}

fn find_bundled_j4rs_jar() -> Option<PathBuf> {
    let cargo_home = cargo_home()?;
    for root in [
        cargo_home.join("registry/src"),
        cargo_home.join("git/checkouts"),
    ] {
        if let Some(found) = find_j4rs_fat_jar(&root, 5) {
            return Some(found);
        }
    }
    None
}

fn cargo_home() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CARGO_HOME") {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo"))
}

fn find_j4rs_fat_jar(root: &Path, depth: usize) -> Option<PathBuf> {
    if depth == 0 || !root.is_dir() {
        return None;
    }
    let mut best: Option<PathBuf> = None;
    let Ok(entries) = fs::read_dir(root) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_file()
            && name.starts_with("j4rs-")
            && name.ends_with("-jar-with-dependencies.jar")
        {
            if best
                .as_ref()
                .is_none_or(|prev| path.file_name() > prev.file_name())
            {
                best = Some(path);
            }
        } else if path.is_dir() && should_descend(&name) {
            if let Some(found) = find_j4rs_fat_jar(&path, depth - 1) {
                if best
                    .as_ref()
                    .is_none_or(|prev| found.file_name() > prev.file_name())
                {
                    best = Some(found);
                }
            }
        }
    }
    best
}

fn should_descend(name: &str) -> bool {
    name.starts_with("j4rs")
        || name == "jassets"
        || name.contains("crates.io")
        || name.starts_with("index.")
}
