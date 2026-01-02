#![recursion_limit = "256"]
use crate::clip::init_embedder;
use crate::database::init_database;
use crate::search::{indexing, web_search_text};
use crate::server_arguments::ServerArguments;
use axum::routing::post;
use axum::{Router, routing::get};
use clap::Parser;
use embed_anything::embeddings::embed::Embedder;
use env_logger::Env;
use log::info;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use surrealdb::{Connection, RecordId, Surreal};
use tokio::sync::Mutex;
use tower_http::services::{ServeDir, ServeFile};

mod clip;
mod database;
pub mod metadata_indexer;
pub mod metadata_provider;
mod search;
mod server_arguments;

#[derive(Debug, Serialize, Deserialize)]
struct DbImage {
    id: RecordId,
    image_path: String,
}

#[derive(Clone)]
pub struct AppState<C>
where C:Connection{
    pub arguments: ServerArguments,
    pub db: Arc<Mutex<Surreal<C>>>,
    pub embedder: Arc<Mutex<Embedder>>,
}

async fn tokio_main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let cla = ServerArguments::parse();

    let static_dir = "target/client/dist";

    let app_state = AppState {
        arguments: cla.clone(),
        db: Arc::new(Mutex::new(init_database(&cla).await.unwrap())),
        embedder: Arc::new(Mutex::new(init_embedder().await.unwrap())),
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
