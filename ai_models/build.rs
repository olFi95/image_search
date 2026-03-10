use burn_import::onnx::ModelGen;
use hf_hub::api::sync::Api;
use std::fs;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let api = Api::new()?;

    #[cfg(feature = "arcface")]
    load_arcface(&api)?;
    #[cfg(feature = "yolo")]
    load_yolo(&api)?;
    #[cfg(feature = "age_gender")]
    load_age_gender(&api)?;
    #[cfg(feature = "clip")]
    {
        load_clip_vision(&api)?;
        load_clip_text(&api)?;
    }
    Ok(())
}

fn load_arcface(api: &Api) -> anyhow::Result<()> {
    let repo = api.model("garavv/arcface-onnx".to_string());
    let downloaded_model = repo.get("arc.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("arc.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(path_to_os_string(upgraded_model)?.as_str())
        .out_dir("arcface")
        .run_from_script();
    copy_if_changed(
        &out_dir.join("arcface").join("arc.bpk"),
        &PathBuf::from("../models/arcface_model.bpk"),
    )?;
    Ok(())
}
fn load_yolo(api: &Api) -> anyhow::Result<()> {
    let repo = api.model("AdamCodd/YOLOv11n-face-detection".to_string());
    let downloaded_model = repo.get("model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("yolo.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;
    ModelGen::new()
        .input(path_to_os_string(upgraded_model)?.as_str())

        .out_dir("yolo")
        .run_from_script();
    copy_if_changed(
        &out_dir.join("yolo").join("yolo.bpk"),
        &PathBuf::from("../models/yolo.bpk"),
    )?;
    Ok(())
}

fn load_age_gender(api: &Api) -> anyhow::Result<()> {
    let repo = api.model("onnx-community/age-gender-prediction-ONNX".to_string());
    let downloaded_model = repo.get("onnx/model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("age_gender.ver16.onnx");

    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(path_to_os_string(upgraded_model)?.as_str())
        .out_dir("age_gender")
        .run_from_script();
    copy_if_changed(
        &out_dir.join("age_gender").join("age_gender.bpk"),
        &PathBuf::from("../models/age_gender.bpk"),
    )?;
    Ok(())
}

fn load_clip_vision(api: &Api) -> anyhow::Result<()> {
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
    copy_if_changed(
        &out_dir.join("clip_vit_large_patch14").join("vision_model.bpk"),
        &PathBuf::from("../models/vision_model.bpk"),
    )?;
    Ok(())
}
fn load_clip_text(api: &Api) -> anyhow::Result<()> {
    let repo = api.model("Xenova/clip-vit-large-patch14".to_string());
    let downloaded_model = repo.get("onnx/text_model.onnx")?;
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    let upgraded_model = out_dir.join("text_model.ver16.onnx");
    println!("cargo:rerun-if-changed={}", downloaded_model.display());

    onnx_updater::init()?;
    onnx_updater::update(&downloaded_model, &upgraded_model)?;

    ModelGen::new()
        .input(upgraded_model.to_str().unwrap())
        .out_dir("clip_vit_large_patch14")
        .run_from_script();
    copy_if_changed(
        &out_dir.join("clip_vit_large_patch14").join("text_model.bpk"),
        &PathBuf::from("../models/text_model.bpk"),
    )?;
    Ok(())
}

fn path_to_os_string(upgraded_model: PathBuf) -> anyhow::Result<String> {
    upgraded_model.into_os_string().into_string()
        .map_err(|err| anyhow::anyhow!("Failed to get path for upgraded model: {:?}", err))
}

fn copy_if_changed(src: &PathBuf, dst: &PathBuf) -> anyhow::Result<()> {
    let src_content = fs::read(src)?;
    if dst.exists() {
        let dst_content = fs::read(dst)?;
        if src_content == dst_content {
            return Ok(());
        }
    }
    if let Some(parent) = dst.parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(dst, src_content)?;
    Ok(())
}

