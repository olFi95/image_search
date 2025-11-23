use crate::AppState;
use crate::database::init_database;
use crate::search::ImageType;
use burn::prelude::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use data::ImagePathResult;
use image::open;
use log::{info, error};
use rand::prelude::SliceRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::collections::HashSet;
use surrealdb::Surreal;
use surrealdb::engine::remote::ws::Client;
use walkdir::WalkDir;
use clip::utils::image_to_resnet;
use tokio::sync::mpsc;

pub async fn clip(state: &AppState, input: String) -> Vec<f32> {
    let clip_embedder = state.embedder.lock().await;
    clip_embedder.embed_text_single(input.as_str()).await
        .expect("Something went wrong embedding text")
}

pub async fn embed_all_images_in_dir(
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let db: Surreal<Client> = init_database(&state.arguments).await?;
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

    // 3-stufige Pipeline mit Channels
    let (tx_new_paths, mut rx_new_paths) = mpsc::channel::<Vec<String>>(3);
    let (tx_prepared, mut rx_prepared) = mpsc::channel::<PreparedBatch>(3);

    // Stufe 1: DB-Check für neue Bilder
    let db_clone = db.clone();
    let db_check_task = tokio::spawn(async move {
        for image_paths in all_image_paths.chunks(image_chunk_size) {
            let mut response = db_clone
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
                image_paths.len(),
                new_paths.len()
            );

            if !new_paths.is_empty() {
                if tx_new_paths.send(new_paths).await.is_err() {
                    break;
                }
            } else {
                info!("No new images to process, skipping chunk.");
            }
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    });

    // Stufe 2: Paralleles Laden und Preprocessing der Bilder
    let preprocessing_task = tokio::spawn(async move {
        while let Some(new_paths) = rx_new_paths.recv().await {
            let prepared_batch = tokio::task::spawn_blocking(move || {
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

                PreparedBatch {
                    image_paths: new_paths,
                    prepared_buffers: all_prepared_image_buffers,
                }
            }).await?;

            if tx_prepared.send(prepared_batch).await.is_err() {
                break;
            }
        }
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
    });

    // Stufe 3: GPU-Embedding und DB-Schreiben
    let mut total_embedded = 0;
    while let Some(batch) = rx_prepared.recv().await {
        if batch.prepared_buffers.is_empty() {
            continue;
        }

        let batch_size = batch.prepared_buffers.len();

        // GPU-Embedding in spawn_blocking ausführen (synchron, um Send-Problem zu vermeiden)
        let model_weights = state.arguments.model_weights.clone();
        let embeddings = tokio::task::spawn_blocking(move || {
            let device = WgpuDevice::DefaultDevice;
            let model = clip::clip_vit_large_patch14::Model::<Wgpu>::from_file(&model_weights, &device);

            let flattened_image_buffers: Vec<f32> = batch.prepared_buffers
                .iter()
                .flatten()
                .cloned()
                .collect();

            let image_data = burn::tensor::TensorData::new(
                flattened_image_buffers,
                [batch_size, 3, 224, 224],
            );
            let image_tensor = Tensor::<Wgpu, 4>::from_data(image_data.convert::<f32>(), &device);

            // GPU-Embedding
            let output = model.forward(image_tensor);

            let data = output.to_data();
            let bytes = data.bytes;
            let float_data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            let embeddings: Vec<Vec<f32>> =
                float_data.chunks(768).map(|chunk| chunk.to_vec()).collect();

            (batch.image_paths, embeddings)
        }).await?;

        let image_types: Vec<ImageType> = embeddings.0
            .into_iter()
            .zip(embeddings.1)
            .map(|(image_path, embedding)| ImageType {
                id: None,
                image_path,
                embedding,
            })
            .collect();

        // DB-Schreiben
        db.insert::<Vec<ImageType>>("image")
            .content(image_types)
            .await?;

        total_embedded += batch_size;
        info!("Embedded and stored {} images (total: {})", batch_size, total_embedded);
    }

    // Warte auf Abschluss aller Tasks
    db_check_task.await??;
    preprocessing_task.await??;

    info!("Done embedding {} images, updating index.", total_embedded);
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

// Hilfsstruct für Pipeline
struct PreparedBatch {
    image_paths: Vec<String>,
    prepared_buffers: Vec<Vec<f32>>,
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
