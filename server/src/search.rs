use crate::clip::clip;
use crate::metadata_indexer::MetadataIndexer;
use crate::metadata_provider::metadata_query_engine::MetadataQueryEngine;
use crate::metadata_provider::model::BaseImage;
use crate::{AppState, DbImage};
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use axum::response::IntoResponse;
use burn::prelude::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use data::{FaceBoundingBox, FacesRequest, FacesResponse, ImageReference, SearchParams, SearchResponse};
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue};

#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct ImageType {
    pub id: Option<RecordId>,
    pub image_path: String,
    pub embedding: Vec<f32>,
}

pub async fn web_search_text<B: Backend>(
    State(state): State<AppState<B>>,
    Json(params): Json<SearchParams>,
) -> Result<Json<SearchResponse>, StatusCode>{
    debug!("Handle Search with params: {:?}", params);

    let db = state.db.lock().await;
    let embedding = clip(&state, params.q.as_str()).await;
    let mut query_vector = embedding.clone();

    info!("image_paths: {:?}", params.referenced_images);
    let media_dir = state
        .arguments
        .shellexpand_media_dir()
        .expect("media dir could not be loaded");
    let media_dir_str = media_dir
        .into_os_string()
        .into_string()
        .expect("media dir could not be converted to string");

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
                r#"
                    SELECT
                        id,
                        path AS image_path,
                        ->has_image_embedding_vector->image_embedding_vector[0].embedding AS embedding
                    FROM base_image
                    WHERE path IN $image_paths
                "#,
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
        LET $similar_vectors = (
            SELECT
                id,
                vector::distance::knn() AS similarity
            FROM image_embedding_vector
            WHERE embedding <|500, 150|> $reference
            ORDER BY similarity ASC
        );


        SELECT
            similarity,
            <-has_image_embedding_vector<-base_image[0].id AS id,
            <-has_image_embedding_vector<-base_image[0].path AS image_path
        FROM $similar_vectors;
    "#;

    let mut response = db
        .query(query)
        .bind(("reference", query_vector))
        .await
        .map_err(|err| {
            tracing::error!("DB query error: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let db_images: Vec<DbImage> = response.take(1).map_err(|err| {
        tracing::error!("Failed to deserialize response: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let images: Vec<ImageReference> = db_images
        .into_iter()
        .map(|img| ImageReference {
            id: img.id_string(),
            image_path: img.image_path.replace(&media_dir_str, "media/"),
        })
        .collect();

    Ok(Json(SearchResponse { images }))
}

pub async fn get_faces<B: Backend>(
    State(state): State<AppState<B>>,
    Json(params): Json<FacesRequest>,
) -> Result<Json<FacesResponse>, StatusCode> {
    debug!("Handle get_faces for image: {:?}", params.image_path);

    let media_dir = state
        .arguments
        .shellexpand_media_dir()
        .expect("media dir could not be loaded");
    let media_dir_str = media_dir
        .into_os_string()
        .into_string()
        .expect("media dir could not be converted to string");

    let absolute_path = if params.image_path.starts_with("media/") {
        params.image_path.replacen("media/", &media_dir_str, 1)
    } else {
        params.image_path.clone()
    };

    debug!("Resolved absolute_path for faces: {:?}", absolute_path);

    let db = state.db.lock().await;

    let mut response = db
        .query("SELECT * FROM base_image WHERE path = $path")
        .bind(("path", absolute_path))
        .await
        .map_err(|err| {
            error!("DB query error: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let base_images: Vec<BaseImage> = response.take(0).map_err(|err| {
        error!("Failed to deserialize base image: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let base_image = match base_images.first() {
        Some(img) => img,
        None => {
            debug!("No base image found for path: {:?}", params.image_path);
            return Ok(Json(FacesResponse { faces: vec![] }));
        }
    };

    let query_engine = MetadataQueryEngine::new(db.clone());
    let metadata = query_engine
        .get_all_metadata_attached_to_base_image(base_image)
        .await
        .map_err(|err| {
            error!("Failed to get metadata: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let faces: Vec<FaceBoundingBox> = metadata
        .faces
        .into_iter()
        .map(|face| {
            let age_gender = face.age_and_gender.first();
            FaceBoundingBox {
                top_left_x: face.top_left_x,
                top_left_y: face.top_left_y,
                bottom_right_x: face.bottom_right_x,
                bottom_right_y: face.bottom_right_y,
                confidence: face.confidence,
                age: age_gender.map(|ag| ag.age),
                gender: age_gender.map(|ag| ag.gender),
            }
        })
        .collect();

    Ok(Json(FacesResponse { faces }))
}

pub async fn indexing<B: Backend>(State(state): State<AppState<B>>) -> impl IntoResponse{
    let state = state.clone();

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();

        rt.block_on(async {
            let device = WgpuDevice::DefaultDevice;

            let metadata_indexer: MetadataIndexer<_, Wgpu<f32, i64>> = MetadataIndexer::new(
                state.db.lock().await.clone(),
                device,
                state.arguments.arcface_model_weights.clone(),
                state.arguments.yolo_model_weights.clone(),
                state.arguments.clip_vision_weights.clone(),
                state.arguments.clip_text_weights.clone(),
                state.arguments.age_and_gender_model_weights.clone(),
            );

            metadata_indexer
                .index_metadata(state.arguments.shellexpand_media_dir().expect("media dir"))
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
