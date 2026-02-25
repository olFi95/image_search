#![recursion_limit = "256"]
use anyhow::Context;
use crate::clip::init_embedder;
use crate::database::init_database;
use crate::search::{indexing, web_search_text};
use crate::server_arguments::ServerArguments;
use axum::routing::post;
use axum::{Router, routing::get};
use clap::Parser;
use embed_anything::embeddings::embed::Embedder;
use log::info;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use surrealdb::Surreal;
use surrealdb::engine::any::Any;
use surrealdb::types::{RecordId, RecordIdKey, SurrealValue};
use tokio::sync::Mutex;
use tower_http::services::{ServeDir, ServeFile};
use tracing_subscriber::EnvFilter;

mod clip;
mod database;
pub mod metadata_indexer;
pub mod metadata_provider;
mod search;
mod server_arguments;

#[derive(Debug, Serialize, SurrealValue, Deserialize)]
struct DbImage {
    id: RecordId,
    image_path: String,
}
impl DbImage {
    pub fn id_string(&self) -> String {
        let key_string = match &self.id.key {
            RecordIdKey::String(s) => s.to_string(),
            RecordIdKey::Number(n) => n.to_string(),
            RecordIdKey::Uuid(u) => u.to_string(),
            RecordIdKey::Array(a) => format!("{a:?}"),
            RecordIdKey::Object(o) => format!("{o:?}"),
            RecordIdKey::Range(r) => format!("{r:?}"),
        };

        format!("{}:{}", self.id.table, key_string)
    }
}

#[derive(Clone)]
pub struct AppState{
    pub arguments: ServerArguments,
    pub db: Arc<Mutex<Surreal<Any>>>,
    pub embedder: Arc<Mutex<Embedder>>,
}

fn init_logging() -> anyhow::Result<()> {
    let default_level = tracing::Level::WARN.to_string();
    let package_log_level = tracing::Level::TRACE.to_string();
    let package_name = env!("CARGO_CRATE_NAME");
    let log_directive = format!("{default_level},{package_name}={package_log_level}");
    // respect RUST_LOG environment variable
    let env_filter = match std::env::var("RUST_LOG") {
        Ok(env) => {
            EnvFilter::builder().parse(env)?
        }
        Err(_) => {
            EnvFilter::builder().parse(log_directive)?
        }
    };
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter(env_filter)
        .init();
    Ok(())
}

async fn tokio_main() -> anyhow::Result<()> {
    init_logging()?;
    let cla = ServerArguments::parse();

    let static_dir = "target/client/dist";

    let surreal_db_client = init_database(&cla).await
        .context("Failed to initialize surreal db connection!")?;
    let embedder = init_embedder().await
        .map_err(anyhow::Error::from_boxed)
        .context("Failed to initialize embedder!")?;
    let app_state = AppState {
        arguments: cla.clone(),
        db: Arc::new(Mutex::new(surreal_db_client)),
        embedder: Arc::new(Mutex::new(embedder)),
    };

    let media_dir = cla.shellexpand_media_dir()?;
    let app = Router::new()
        .route("/search", post(web_search_text))
        .route("/scan", get(indexing))
        .with_state(app_state)
        .nest_service("/media", ServeDir::new(&media_dir))
        .fallback_service(
            ServeDir::new(static_dir)
                .not_found_service(ServeFile::new(format!("{}/index.html", static_dir))),
        );
    info!("HTTP server läuft auf http://{}", cla.get_socket_addr());

    axum_server::bind(cla.get_socket_addr())
        .serve(app.into_make_service())
        .await?;
    Ok(())
}

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        // increase thread stack size to 32 MB to prevent stack overflow when embedding images with WGPU
        .thread_stack_size(32 * 1024 * 1024)
        .build()
        .unwrap()
        .block_on(async {
            tokio_main().await.expect("Tokio main failed");
        })
}
