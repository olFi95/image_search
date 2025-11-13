use crate::clip::{clip, embed_all_images_in_dir};
use crate::{AppState, DbImage};
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::{Router, debug_handler, response::IntoResponse, routing::get};
use data::{ImageReference, ImageReferenceEmbedding, SearchParams, SearchResponse};
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use surrealdb::{Error, RecordId};
use tokio::runtime::Builder;
use tracing::field::debug;
use walkdir::WalkDir;

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

    trace!("image_paths: {:?}", params.referenced_images);
    if params.referenced_images.len() > 0 {
        let image_paths: Vec<String> = params
            .referenced_images
            .into_iter()
            .filter(|img| img.starts_with("media/"))
            .map(|img| img.replacen("media/", state.arguments.media_dir.as_str(), 1))
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
        if marked_image.len() > 0 {
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
            image_path: img.image_path.replace(&state.arguments.media_dir, "media/"),
        })
        .collect();

    Ok(Json(SearchResponse { images }))
}

#[debug_handler]
pub async fn web_scan(State(state): State<AppState>) -> impl IntoResponse {
    let state_cloned = state.clone();

    Builder::new_multi_thread()
        .thread_name("embedder-task")
        .thread_stack_size(32 * 1024 * 1024) // 32 MB stack
        .enable_all()
        .build()
        .unwrap()
        .spawn(async move {
            let result = embed_all_images_in_dir(&state_cloned).await;
            match result {
                Ok(_) => info!("embedded all images successfully."),
                Err(e) => {
                    error!("Error embedding images: {}", e);
                }
            }
        })
        .await
        .unwrap();

    StatusCode::OK
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
        let result = average_slices(&vec![a.as_slice(), b.as_slice()]);
        assert_eq!(result, vec![1.0, 1.5, 3.0, 4.0, 5.0]);
    }
}
