use std::fs;
use burn_import::onnx::ModelGen;
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Download ONNX model from Hugging Face
    let api = Api::new()?;
    let repo = api.model("Xenova/clip-vit-large-patch14".to_string());
    let downloaded_model = repo.get("onnx/vision_model.onnx")?;

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    // Step 2: Set up Python venv under ./target/venv
    let venv_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set")).join("venv");
    if !venv_dir.exists() {
        println!("cargo:warning=Creating Python venv...");
        let python = which::which("python3")?;
        let status = Command::new(python)
            .arg("-m")
            .arg("venv")
            .arg(&venv_dir)
            .status()?;
        if !status.success() {
            panic!("Failed to create Python virtual environment");
        }
    }

    // Step 3: Install Python dependencies into venv
    let pip_path = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts/pip.exe")
    } else {
        venv_dir.join("bin/pip")
    };

    println!("cargo:warning=Installing Python dependencies...");
    let status = Command::new(&pip_path)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status()?;
    if !status.success() {
        panic!("Failed to upgrade pip");
    }

    let status = Command::new(&pip_path)
        .args(["install", "onnx", "onnx_graphsurgeon"])
        .status()?;
    if !status.success() {
        panic!("Failed to install required Python packages");
    }

    // Step 4: Run Python script to convert opset to 16
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("vision_model.ver16.onnx");

    let python_path = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts/python.exe")
    } else {
        venv_dir.join("bin/python3")
    };

    println!("Running opset upgrade script...");
    let status = Command::new(&python_path)
        .arg("scripts/upgrade_opset.py")
        .arg(downloaded_model.as_os_str())
        .arg(upgraded_model.as_os_str())
        .status()?;

    if !status.success() {
        panic!("Opset upgrade script to upgrade ONNX model failed.");
    }

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("clip_vit_large_patch14")
        .run_from_script();
    fs::copy(out_dir.join("clip_vit_large_patch14").join("vision_model.mpk"), PathBuf::from("../models/vision_model.mpk"))?;
    Ok(())
}
