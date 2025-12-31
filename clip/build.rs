use burn_import::onnx::ModelGen;
use hf_hub::api::sync::Api;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("Xenova/clip-vit-large-patch14".to_string());
    let downloaded_model = repo.get("onnx/vision_model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("vision_model.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("clip_vit_large_patch14")
        .run_from_script();
    let dest_dir = PathBuf::from("../models");
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)?;
    }
    fs::copy(
        out_dir
            .join("clip_vit_large_patch14")
            .join("vision_model.bpk"),
        PathBuf::from("../models/vision_model.bpk"),
    )?;
    Ok(())
}
