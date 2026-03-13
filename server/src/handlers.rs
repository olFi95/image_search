use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use burn::prelude::Backend;
use log::error;
use data::{FacesRequest, FacesResponse, DatabaseStatusResponse, SearchParams, SearchResponse};
use crate::AppState;

pub async fn web_search_text<B: Backend>(
    State(state): State<AppState<B>>,
    Json(params): Json<SearchParams>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let media_dir = state
        .arguments
        .shellexpand_media_dir()
        .map_err(|err| {
            error!("Failed to get media dir: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let media_dir_str = media_dir
        .into_os_string()
        .into_string()
        .map_err(|_| {
            error!("Failed to convert media dir to string");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Berechne Text-Embedding
    let clip_embedder = state.clip_embedder.lock().await;
    let text_embedding = clip_embedder.embed_text(&[params.q.as_str()])
        .into_iter()
        .next()
        .unwrap_or_default();
    drop(clip_embedder); // Gebe Lock frei

    let result = state.query_service
        .search_text(&state.db, text_embedding, params, &media_dir_str)
        .await
        .map_err(|err| {
            error!("Search failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(result))
}

pub async fn web_get_faces<B: Backend>(
    State(state): State<AppState<B>>,
    Json(params): Json<FacesRequest>,
) -> Result<Json<FacesResponse>, StatusCode> {
    let media_dir = state
        .arguments
        .shellexpand_media_dir()
        .map_err(|err| {
            error!("Failed to get media dir: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    let media_dir_str = media_dir
        .into_os_string()
        .into_string()
        .map_err(|_| {
            error!("Failed to convert media dir to string");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let result = state.query_service
        .get_faces(&state.db, params, &media_dir_str)
        .await
        .map_err(|err| {
            error!("Get faces failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(result))
}

pub async fn web_get_database_status<B: Backend>(
    State(state): State<AppState<B>>,
) -> Result<Json<DatabaseStatusResponse>, StatusCode> {
    let result = state.query_service
        .get_database_status(&state.db)
        .await
        .map_err(|err| {
            error!("Get number of images failed: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(result))
}



