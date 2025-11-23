use futures::stream::iter;
use crate::AppState;
use crate::database::init_database;
use crate::search::ImageType;
use burn::prelude::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use data::ImagePathResult;
use image::{DynamicImage, open};
use log::{info, error};
use rand::prelude::SliceRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::io::Cursor;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use futures::StreamExt;
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::Client;
use tracing::field::debug;
use walkdir::WalkDir;
use clip::utils::image_to_resnet;
use picture_preprocessing::preprocessor::Preprocessor;

pub async fn clip(state: &AppState, input: String) -> Vec<f32> {
    let clip_embedder = state.embedder.lock().await;
    clip_embedder.embed_text_single(input.as_str()).await
        .expect("Something went wrong embedding text")
}

pub async fn embed_all_images_in_dir(
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db: Surreal<Client> = init_database(&state.arguments).await?;
    let device = WgpuDevice::DefaultDevice;
    let model =
        clip::clip_vit_large_patch14::Model::from_file(state.arguments.model_weights.as_str(), &device);
    let media_dir = state.arguments.shellexpand_media_dir()?;
    info!("Searching directory {media_dir:?}.");
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
        if new_paths.len()==0 {
            debug("No new images to process, skipping chunk.");
            continue;
        }
        // let preprocessor = Preprocessor::new().await;
        // let mut all_prepared_image_buffers: Vec<Vec<f32>> = vec![];
        // for new_path in &new_paths {
        //     match open(&new_path) {
        //         Ok(img) => {
        //             let prepared = preprocessor.preprocess(&img).await;
        //             all_prepared_image_buffers.push(prepared);
        //         }
        //         Err(err) => {
        //             error!("Failed to open image {}: {}", new_path, err);
        //         }
        //     }
        // }

        let all_prepared_image_buffers: Vec<Vec<f32>> = new_paths
            .par_iter()
            .filter_map(|image_path| match open(image_path) {
                Ok(img) => {
                    let prepared = image_to_resnet(img);
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

    info!("Done embedding images, updating index.");
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
