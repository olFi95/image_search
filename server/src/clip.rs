use crate::AppState;
use embed_anything::embeddings::embed::Embedder;
use log::{error, info};
use std::path::PathBuf;
use surrealdb::Connection;
use walkdir::WalkDir;

pub async fn clip<C>(state: &AppState<C>, input: String) -> Vec<f32>
    where C: Connection {

    let clip_embedder = state.embedder.lock().await;
    let embedding_result = &clip_embedder.embed(&[&input], None, None).await.unwrap()[0];
    embedding_result.to_dense().unwrap()
}

pub async fn init_embedder() -> Result<Embedder, Box<dyn std::error::Error + Send + Sync>> {
    let clip_embedder =
        Embedder::from_pretrained_hf("Clip", "openai/clip-vit-large-patch14", None, None, None)?;
    info!("Embedder initialized");
    Ok(clip_embedder)
}

pub fn get_all_directories_in_dir(media_dir: &PathBuf) -> Vec<String> {
    let all_image_paths: Vec<String> = WalkDir::new(&media_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| {
            {
                // permission errors are encountered here
                entry.inspect_err(|error| error!("Image load error: {:?}", error))
            }
            .ok()
        })
        .filter(|e| e.metadata().is_ok_and(|file| file.is_file()))
        .filter(|e| {
            e.path().extension().is_some_and(|ext| {
                matches!(
                    ext.to_str().unwrap_or("").to_lowercase().as_str(),
                    "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
                )
            })
        })
        .map(|e| e.path().display().to_string())
        .collect();
    all_image_paths
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::path::Path;

    #[test]
    fn test_matches() {
        assert!(!matches!(
            Path::new("file.txt")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(matches!(
            Path::new("file.jpg")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(matches!(
            Path::new("file.png")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(!matches!(
            Path::new("file.mp4")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
        assert!(!matches!(
            Path::new("file")
                .extension()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap()
                .to_lowercase()
                .as_str(),
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" | "tiff"
        ));
    }
}
