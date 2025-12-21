use crate::clip::{clip, embed_all_images_in_dir, embed_faces};
use crate::{AppState, DbImage};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::{debug_handler, response::IntoResponse};
use data::{ImageReference, SearchParams, SearchResponse};
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;
use tokio::join;
use tokio::runtime::Handle;

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

#[debug_handler]
pub async fn web_scan(State(state): State<AppState>) -> impl IntoResponse {
    let state_cloned = state.clone();

    // embed_faces ist !Send → spawn_blocking benutzen
    let face_result = tokio::task::spawn_blocking(move || {
        // block_on, um async embed_faces zu warten
        tokio::runtime::Handle::current().block_on(embed_faces(&state_cloned))
    })
        .await
        .unwrap_or_else(|e| Err(Box::<dyn std::error::Error + Send + Sync>::from(e)));

    match face_result {
        Ok(_) => info!("embedded all faces successfully."),
        Err(e) => error!("Error embedding faces: {}", e),
    }

    // embed_all_images_in_dir ist Send → kann normal awaited werden
    // let state_cloned = state.clone();
    // let image_result = embed_all_images_in_dir(&state_cloned).await;
    // match image_result {
    //     Ok(_) => info!("embedded all images successfully."),
    //     Err(e) => error!("Error embedding images: {}", e),
    // }

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
        let result = average_slices(&vec![&a, &b]);
        assert_eq!(result, vec![1.0, 1.5, 3.0, 4.0, 5.0]);
    }
}
