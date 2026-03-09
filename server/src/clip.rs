use crate::AppState;
use burn::prelude::Backend;
use log::error;
use std::path::PathBuf;
use walkdir::WalkDir;

pub fn clip<B: Backend>(state: &AppState<B>, input: &str) -> Vec<f32> {
    let clip_embedder = state.clip_embedder.blocking_lock();
    clip_embedder.embed_text(&[input]).into_iter().next().unwrap_or_default()
}

pub fn get_all_directories_in_dir(media_dir: &PathBuf) -> Vec<String> {
    let all_image_paths: Vec<String> = WalkDir::new(media_dir)
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
