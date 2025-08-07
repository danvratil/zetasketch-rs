use j4rs::{JvmBuilder, MavenArtifact};

fn main() {
    let jvm = JvmBuilder::new().build().expect("Failed to create JVM");
    jvm.deploy_artifact(&MavenArtifact::from(
        "com.google.zetasketch:zetasketch:0.1.0",
    ))
    .expect("Failed to deploy zetasketch Maven artifact");
    jvm.deploy_artifact(&MavenArtifact::from("it.unimi.dsi:fastutil:8.2.2"))
        .expect("Failed to deploy fastutil Maven artifact");
}
