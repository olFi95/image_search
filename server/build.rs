use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=../client");
    let dist_dir = PathBuf::from("../target").join("client").join("dist");
    let status = Command::new("trunk")
        .current_dir("../client") // Wechsle in das Verzeichnis des `web`-Projekts
        .arg("build")
        .arg("--release")
        .arg("-d")
        .arg(dist_dir.as_os_str().to_str().unwrap())
        .status()
        .expect("Failed to compile the Web project");
    if !status.success() {
        panic!("Could not build client:{:?}", status)
    }

    Ok(())
}