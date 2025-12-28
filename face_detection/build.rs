use std::error::Error;
use std::fs;
use burn_import::onnx::ModelGen;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;

    load_arcface(&api)?;
    load_yolo(&api)?;
    load_age_gender(&api)?;
    Ok(())
}

fn load_arcface(api: &Api) -> Result<(), Box<dyn Error>> {
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
    fs::copy(out_dir.join("arcface").join("arc.bpk"), PathBuf::from("../models/arcface_model.bpk"))?;
    Ok(())
}
fn load_yolo(api: &Api) -> Result<(), Box<dyn Error>> {
    let repo = api.model("AdamCodd/YOLOv11n-face-detection".to_string());
    let downloaded_model = repo.get("model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("yolo.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("yolo")
        .run_from_script();
    let dest_dir = PathBuf::from("../models");
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)?;
    }
    fs::copy(out_dir.join("yolo").join("yolo.bpk"), PathBuf::from("../models/yolo.bpk"))?;
    Ok(())
}
fn load_age_gender(api: &Api) -> Result<(), Box<dyn Error>> {
    let repo = api.model("onnx-community/age-gender-prediction-ONNX".to_string());
    let downloaded_model = repo.get("onnx/model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("age_gender.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("age_gender")
        .run_from_script();
    let dest_dir = PathBuf::from("../models");
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)?;
    }
    fs::copy(out_dir.join("age_gender").join("age_gender.bpk"), PathBuf::from("../models/age_gender.bpk"))?;
    Ok(())
}
