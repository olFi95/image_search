use crate::metadata_indexer::MetadataIndexer;
use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
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



pub async fn indexing<B: Backend>(State(state): State<AppState<B>>) -> impl IntoResponse{
    let state = state.clone();

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();

        rt.block_on(async {
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
                .index_metadata(state.arguments.shellexpand_media_dir().expect("media dir"))
                .await
                .expect("indexing failed");
        });
    });

    StatusCode::ACCEPTED
}

