use std::fs;
use burn_import::onnx::ModelGen;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("garavv/arcface-onnx".to_string());
    let downloaded_model = repo.get("arc.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("arc.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("arcface")
        .run_from_script();
    let dest_dir = PathBuf::from("../models");
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)?;
    }
    fs::copy(out_dir.join("arcface").join("arc.mpk"), PathBuf::from("../models/arcface_model.mpk"))?;
    Ok(())
}
