#![recursion_limit = "256"]
use crate::clip::init_embedder;
use crate::database::init_database;
use crate::search::{web_scan, web_search_text};
use crate::server_arguments::ServerArguments;
use axum::extract::Query;
use axum::routing::{get_service, post, trace};
use axum::{Json, Router, routing::get};
use clap::Parser;
use data::{ImageReference, SearchParams, SearchResponse};
use embed_anything::embeddings::embed::Embedder;
use log::{info, trace};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::{Arc, LazyLock};
use surrealdb::engine::remote::ws::{Client, Ws};
use surrealdb::opt::auth::Root;
use surrealdb::syn::token::Keyword::Search;
use surrealdb::{RecordId, Surreal};
use tokio::sync::Mutex;
use tonic::{Request, Response, Status, transport::Server};
use tonic_web::GrpcWebLayer;
use tower_http::services::{ServeDir, ServeFile};

mod clip;
mod database;
mod search;
mod server_arguments;

#[derive(Debug, Serialize, Deserialize)]
struct DbImage {
    id: RecordId,
    image_path: String,
}

#[derive(Clone)]
pub struct AppState {
    pub arguments: ServerArguments,
    pub db: Arc<Mutex<Surreal<Client>>>,
    pub embedder: Arc<Mutex<Embedder>>,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let cla = ServerArguments::parse();

    let static_dir = "target/client/dist";

    let app_state = AppState {
        arguments: cla.clone(),
        db: Arc::new(Mutex::new(init_database(&cla).await.unwrap())),
        embedder: Arc::new(Mutex::new(init_embedder().await.unwrap())),
    };

    let app = Router::new()
        .route("/search", post(web_search_text))
        .route("/scan", get(web_scan))
        .with_state(app_state)
        .nest_service("/media", ServeDir::new(&cla.media_dir))
        .fallback_service(
            ServeDir::new(static_dir)
                .not_found_service(ServeFile::new(format!("{}/index.html", static_dir))),
        );
    info!("HTTP server läuft auf http://{}", cla.get_socket_addr());

    axum_server::bind(cla.get_socket_addr())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
