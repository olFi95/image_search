use std::path::PathBuf;
use image::{open, DynamicImage, ImageResult};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;
use surrealdb::sql::Thing;
use crate::clip::ImageFaceEmbeddingCheckResult;

pub struct BaseImage {
    pub id: Option<RecordId>,
    pub path: String,
    pub image: DynamicImage,
}

impl BaseImage {
    pub(crate) fn open(image_path: PathBuf) -> Self {
        let image = open(&image_path).expect("Failed to open image");
        Self {
            id: None,
            path: image_path.into_os_string().to_str().expect("cannot convert provided path to string").to_string(),
            image,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Metadata<M> {
    pub id: Option<String>,
    pub metadata: Option<M>,
    pub base: Option<RecordId>,
}


pub trait MetadataProvider<B, M> {
    fn extract(&self, base_data_elements: &Vec<B>) -> anyhow::Result<Vec<Metadata<M>>>;
}