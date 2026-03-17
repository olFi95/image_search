use crate::metadata_indexer::MetadataIndexer;
use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use axum::response::IntoResponse;
use burn::prelude::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use serde::{Deserialize, Serialize};
use surrealdb::types::{RecordId, SurrealValue};

#[derive(Debug, Serialize, SurrealValue, Deserialize, Clone)]
pub struct ImageType {
    pub id: Option<RecordId>,
    pub image_path: String,
    pub embedding: Vec<f32>,
}

#[derive(PartialEq, Clone, Serialize)]
pub enum IndexingStatus {
    Idle,
    Started,
    InProgress(IndexState),
    Completed,
}

#[derive(PartialEq, Clone, Serialize)]
pub struct IndexState {
    pub(crate) total: u32,
    pub(crate) progress: u32,
}

pub async fn indexing<B: Backend>(State(state): State<AppState<B>>) -> impl IntoResponse {
    let mut state = state.clone();
    let guard = state.indexing_status.lock().await;

    if matches!(
        *guard,
        IndexingStatus::Started | IndexingStatus::InProgress(_)
    ) {
        return (StatusCode::OK, Json((*guard).clone()));
    }
    drop(guard);
    let indexing_status = state.indexing_status.clone();
    tokio::spawn(async move {
        let device = WgpuDevice::DefaultDevice;

        let metadata_indexer: MetadataIndexer<_, Wgpu<f32, i64>> = MetadataIndexer::new(
            state.db.clone(),
            device,
            state.arguments.arcface_model_weights.clone(),
            state.arguments.yolo_model_weights.clone(),
            state.arguments.clip_vision_weights.clone(),
            state.arguments.clip_text_weights.clone(),
            state.arguments.age_and_gender_model_weights.clone(),
        );

        metadata_indexer
            .index_metadata(state.arguments.shellexpand_media_dir().expect("media dir"), indexing_status.clone())
            .await
            .expect("indexing failed");
        let mut guard = indexing_status.lock_owned().await;
        *guard = IndexingStatus::Completed;
    });

    (StatusCode::ACCEPTED, Json(IndexingStatus::Started))
}
