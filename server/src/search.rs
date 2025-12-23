use std::sync::Arc;
use std::sync::mpsc::channel;
use crate::clip::{clip, embed_all_images_in_dir, embed_faces};
use crate::{AppState, DbImage};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::{debug_handler, response::IntoResponse};
use burn::tensor::Device;
use burn_wgpu::WgpuDevice;
use data::{ImageReference, SearchParams, SearchResponse};
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;
use tokio::join;
use tokio::runtime::Handle;
use crate::metadata_indexer::MetadataIndexer;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageType {
    pub id: Option<RecordId>,
    pub image_path: String,
    pub embedding: Vec<f32>,
}

pub async fn web_search_text(
    State(state): State<AppState>,
    Json(params): Json<SearchParams>,
) -> Result<Json<SearchResponse>, StatusCode> {
    debug!("Handle Search with params: {:?}", params);

    let db = state.db.lock().await; // oder wie du deine DB-Instanz nutzt
    let embedding = clip(&state, params.q).await;
    let mut query_vector = embedding.clone();

    info!("image_paths: {:?}", params.referenced_images);
    let media_dir = state.arguments.shellexpand_media_dir().expect("media dir could not be loaded");
    let media_dir_str = media_dir.into_os_string().into_string().expect("media dir could not be converted to string");

    if !params.referenced_images.is_empty() {
        let image_paths: Vec<String> = params
            .referenced_images
            .into_iter()
            .filter(|img| img.starts_with("media/"))
            .map(|img| img.replacen("media/", &media_dir_str, 1))
            .collect::<Vec<String>>();
        trace!("image_paths: {image_paths:?}");

        let mut marked_image_embeddings_response = db
            .query(
                "
        SELECT id, image_path, embedding FROM image WHERE image_path IN $image_paths",
            )
            .bind(("image_paths", image_paths))
            .await
            .map_err(|err| {
                error!("DB query error: {:?}", err);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        let marked_image: Vec<ImageType> =
            marked_image_embeddings_response.take(0).map_err(|err| {
                error!("Failed to deserialize response: {:?}", err);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        debug!("marked_image_embeddings {}", marked_image.len());
        if !marked_image.is_empty() {
            let slices = marked_image
                .iter()
                .map(|embedding| &embedding.embedding)
                .collect::<Vec<&Vec<f32>>>();
            let selected_images_average = average_slices(&slices);
            query_vector = average_slices(&vec![&selected_images_average, &embedding]);
        }
    }

    let query = r#"
        SELECT
            id,
            image_path,
            vector::distance::knn() AS similarity
        FROM image
        WHERE embedding <| 1000 |> $reference;
    "#;

    let mut response = db
        .query(query)
        .bind(("reference", query_vector))
        .await
        .map_err(|err| {
            tracing::error!("DB query error: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let db_images: Vec<DbImage> = response.take(0).map_err(|err| {
        tracing::error!("Failed to deserialize response: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let images: Vec<ImageReference> = db_images
        .into_iter()
        .map(|img| ImageReference {
            id: img.id.to_string(),
            image_path: img.image_path.replace(&media_dir_str, "media/"),
        })
        .collect();

    Ok(Json(SearchResponse { images }))
}

#[axum::debug_handler]
pub async fn indexing(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let state = state.clone();

    tokio::task::spawn_blocking(move || {
        // ❗ alles Nicht-Send hier rein
        let rt = tokio::runtime::Handle::current();

        rt.block_on(async {
            let device = Arc::new(Box::new(WgpuDevice::DefaultDevice));

            let metadata_indexer = MetadataIndexer::new(
                state.db.lock().await.clone(),
                device,
                state.arguments.arcface_model_weights.clone(),
                state.arguments.yolo_model_weights.clone(),
            );

            metadata_indexer
                .index_metadata(
                    state.arguments
                        .shellexpand_media_dir()
                        .expect("media dir"),
                )
                .await
                .expect("indexing failed");
        });
    });

    StatusCode::ACCEPTED
}
fn average_slices(vectors: &Vec<&Vec<f32>>) -> Vec<f32> {
    assert!(!vectors.is_empty(), "Input must not be empty");

    let len = vectors[0].len();
    assert!(
        vectors.iter().all(|v| v.len() == len),
        "All vectors must have the same length"
    );

    let mut result = vec![0.0; len];

    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            result[i] += val;
        }
    }

    let count = vectors.len() as f32;
    for val in &mut result {
        *val /= count;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tes_average_vector() {
        let a = vec![1.0, 2.0, 4.0, 4.0, 10.0];
        let b = vec![1.0, 1.0, 2.0, 4.0, 0.0];
        let result = average_slices(&vec![&a, &b]);
        assert_eq!(result, vec![1.0, 1.5, 3.0, 4.0, 5.0]);
    }
}
