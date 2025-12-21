use crate::database::init_database;
use crate::search::ImageType;
use crate::AppState;
use burn::prelude::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use data::ImagePathResult;
use embed_anything::embeddings::embed::{EmbedImage, Embedder};
use face_detection::arcface;
use face_detection::face_detector::FaceDetector;
use face_detection::face_embedder::FaceEmbedder;
use image::{open, DynamicImage};
use log::{error, info};
use rand::prelude::SliceRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use surrealdb::engine::remote::ws::Client;
use surrealdb::sql::Thing;
use surrealdb::Surreal;
use walkdir::WalkDir;

pub async fn clip(state: &AppState, input: String) -> Vec<f32> {
    let clip_embedder = state.embedder.lock().await;
    let embedding_result = &clip_embedder.embed(&[&input], None, None).await.unwrap()[0];
    embedding_result.to_dense().unwrap()
}

pub async fn clip_image_path(
    state: &AppState,
    image: PathBuf,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let clip_embedder = state.embedder.lock().await;
    let embedding_result = clip_embedder.embed_image(image, None).await?;
    let embedding_data = embedding_result.embedding;
    Ok(embedding_data.to_dense()?)
}

pub async fn init_embedder() -> Result<Embedder, Box<dyn std::error::Error + Send + Sync>> {
    let clip_embedder =
        Embedder::from_pretrained_hf("Clip", "openai/clip-vit-large-patch14", None, None, None)?;
    info!("Embedder initialized");
    Ok(clip_embedder)
}


pub async fn embed_faces(
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db: Surreal<Client> = init_database(&state.arguments).await?;
    let device = Arc::new(WgpuDevice::DefaultDevice);
    let media_dir = state.arguments.shellexpand_media_dir()?;
    let image_chunk_size = state.arguments.image_chunk_size;

    let face_detector = FaceDetector::new(&state.arguments.yolo_model_weights, device.clone());
    let face_embedder = FaceEmbedder::new(&state.arguments.arcface_model_weights, device.clone());

    let mut all_image_paths = get_all_directories_in_dir(&media_dir);
    all_image_paths.shuffle(&mut rand::rng());

    for image_path in all_image_paths {
        let image_path = image_path.clone();
        let mut response = db
            .query("SELECT id, face_detection_done FROM image WHERE image_path IS $path")
            .bind(("path", image_path.clone()))
            .await?;


        let result = response.take::<Vec<ImageFaceEmbeddingCheckResult>>(0)?;
        if result.len() != 1 {
            continue;
        }
        let element = &result[0];
        if element.face_detection_done.unwrap_or(false) {
            continue;
        }
        let image = open(&image_path)?;
        let faces = face_detector.detect(&image);
        let mut face_embeddings: Vec<FaceEmbedding> = Vec::new();
        for face in faces{
            let embedding = face_embedder.embed(face.face_image);

            let face_embedding = FaceEmbedding{
                id: None,
                score: face.bbox.score,
                bbox: [face.bbox.xmin, face.bbox.ymin, face.bbox.xmax - face.bbox.xmin, face.bbox.ymax - face.bbox.ymin],
                embedding,
            };
            face_embeddings.push(face_embedding);
        }
        let element_id = element.id.clone();
        db.insert::<Vec<FaceEmbedding>>("face_embedding")
            .content(face_embeddings)
            .await?;
        db.query("UPDATE image SET face_detection_done = true WHERE id = $id")
            .bind(("id", element_id))
            .await?;
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageFaceEmbeddingCheckResult {
    pub id: Thing,
    pub face_detection_done: Option<bool>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceEmbedding {
    pub id: Option<Thing>,
    pub score: f32, // confidence score of the face detection
    pub bbox: [f32; 4], // x, y, width, height
    pub embedding: Vec<f32>,
}


pub async fn embed_all_images_in_dir(
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db: Surreal<Client> = init_database(&state.arguments).await?;
    let device = WgpuDevice::DefaultDevice;
    let model: clip::clip_vit_large_patch14::Model<Wgpu> =
        clip::clip_vit_large_patch14::Model::from_file(state.arguments.clip_model_weights.as_str(), &device);
    let media_dir = state.arguments.shellexpand_media_dir()?;
    info!("Searching directory {media_dir:?}.");
    let mut all_image_paths = get_all_directories_in_dir(&media_dir);
    all_image_paths.shuffle(&mut rand::rng());
    let num_images = all_image_paths.clone().len();
    info!("Found {num_images} images in directory.");
    let image_chunk_size = state.arguments.image_chunk_size;
    for image_paths in all_image_paths.chunks(image_chunk_size) {
        let mut response = db
            .query("SELECT image_path FROM image WHERE image_path IN $paths")
            .bind(("paths", image_paths.to_vec()))
            .await?;
        let existing_paths: HashSet<String> = response
            .take::<Vec<ImagePathResult>>(0)?
            .into_iter()
            .map(|img| img.image_path)
            .collect();
        let new_paths: Vec<String> = image_paths
            .iter()
            .filter(|p| !existing_paths.contains(p.as_str()))
            .cloned()
            .collect();
        info!(
            "Found {} images in chunk of which are {} new",
            &image_paths.len(),
            new_paths.len()
        );

        let all_prepared_image_buffers: Vec<Vec<f32>> = new_paths
            .par_iter()
            .filter_map(|image_path| match open(image_path) {
                Ok(img) => {
                    let prepared = image_prepare_resnet(img);
                    Some(prepared)
                }
                Err(err) => {
                    error!("Failed to open image {}: {}", image_path, err);
                    None
                }
            })
            .collect();
        let flattened_image_buffers: Vec<f32> = all_prepared_image_buffers
            .iter()
            .flatten()
            .cloned()
            .collect();
        let image_data = burn::tensor::TensorData::new(
            flattened_image_buffers,
            [all_prepared_image_buffers.len(), 3, 224, 224],
        );
        let image_tensor = Tensor::<Wgpu, 4>::from_data(image_data.convert::<f32>(), &device);
        let image_tensor_clone = image_tensor.clone();
        let output = model.forward(image_tensor_clone);

        let data = output.to_data();
        let bytes = data.bytes;
        let float_data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let embeddings: Vec<Vec<f32>> =
            float_data.chunks(768).map(|chunk| chunk.to_vec()).collect();
        let image_types: Vec<ImageType> = new_paths
            .iter()
            .cloned()
            .zip(embeddings)
            .map(|(image_path, embedding)| ImageType {
                id: None,
                image_path,
                embedding,
            })
            .collect();
        db.insert::<Vec<ImageType>>("image")
            .content(image_types)
            .await?;
    }

    let index_update_result = db.query(
        "DEFINE INDEX IF NOT EXISTS mt_pts ON image FIELDS embedding MTREE DIMENSION 768 DIST COSINE TYPE F32;")
        .query("
        REBUILD INDEX IF EXISTS mt_pts ON image;").await;
    match index_update_result {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("Failed to update index: {}", e);
            Err(e.into())
        }
    }
}

fn get_all_directories_in_dir(media_dir: &PathBuf) -> Vec<String> {
    let mut all_image_paths: Vec<String> = WalkDir::new(&media_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| {
            // permission errors are encountered here
            entry.inspect_err(|error| error!("Image load error: {:?}", error))
        }.ok())
        .filter(|e| e.metadata().is_ok_and(|file| file.is_file()))
        .filter(|e| {
            e.path().extension().is_some_and(|ext| {
                matches!(
                    ext.to_str().unwrap_or("").to_lowercase().as_str(),
                    "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
                )
            })
        })
        .map(|e| e.path().display().to_string())
        .collect();
    all_image_paths
}

fn image_prepare_resnet(img: DynamicImage) -> Vec<f32> {
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let pixels = rgb.as_raw().as_slice(); // &[u8] slice in RGBRGBRGB...

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    // Output in CHW format: [C][H][W]
    let mut data = vec![0.0f32; 3 * 224 * 224];

    for i in 0..(224 * 224) {
        let r = pixels[i * 3] as f32 / 255.0;
        let g = pixels[i * 3 + 1] as f32 / 255.0;
        let b = pixels[i * 3 + 2] as f32 / 255.0;

        data[i] = (r - mean[0]) / std[0]; // Red channel
        data[224 * 224 + i] = (g - mean[1]) / std[1]; // Green channel
        data[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; // Blue channel
    }

    data
}
#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::Path;

    #[test]
    fn test_matches() {
        assert!(!matches!(
            Path::new("file.txt")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(matches!(
            Path::new("file.jpg")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(matches!(
            Path::new("file.png")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(!matches!(
            Path::new("file.mp4")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(!matches!(
            Path::new("file")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
    }
}
