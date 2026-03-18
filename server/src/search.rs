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
    InProgress(IndexState),
}

#[derive(PartialEq, Clone, Serialize)]
pub struct IndexState {
    pub(crate) total: u32,
    pub(crate) already_indexed: u32,
    pub(crate) indexed: u32,
}

pub async fn get_indexing<B: Backend>(State(state): State<AppState<B>>) -> impl IntoResponse {
    let state = state.clone();

    let indexing_status = state.indexing_status.lock().await;
    match *indexing_status {
        IndexingStatus::Idle => {
            Json(IndexingStatus::Idle)
        }
        IndexingStatus::InProgress(ref index_state) => {
            Json(IndexingStatus::InProgress(index_state.clone()))
        }
    }
}
pub async fn start_indexing<B: Backend>(State(state): State<AppState<B>>) -> impl IntoResponse {
    let state = state.clone();

    match *state.indexing_status.lock().await {
        IndexingStatus::Idle => {
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
                *guard = IndexingStatus::Idle;
            });

            StatusCode::ACCEPTED

        }
        IndexingStatus::InProgress(_) => {
            StatusCode::PROCESSING
        }
    }
}
