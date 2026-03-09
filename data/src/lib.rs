use serde::{Deserialize, Serialize};
use urlencoding::encode;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchParams {
    pub q: String,
    #[serde(default)]
    pub referenced_images: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResponse {
    pub images: Vec<ImageReference>,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageReference {
    pub id: String,
    pub image_path: String,
}
impl ImageReference {
    pub fn new(image_path: String) -> Self {
        Self {
            id: encode(&image_path).parse().unwrap(),
            image_path,
        }
    }
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageReferenceEmbedding {
    pub id: String,
    pub image_path: String,
    pub embedding: Vec<f32>,
}
impl ImageReferenceEmbedding {
    pub fn new(image_path: String, embedding: Vec<f32>) -> Self {
        Self {
            id: encode(&image_path).parse().unwrap(),
            image_path,
            embedding,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageReferenceScore {
    pub id: String,
    pub image_path: String,
    pub score: f32,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImagePathResult {
    pub image_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FacesRequest {
    pub image_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FaceBoundingBox {
    pub top_left_x: f32,
    pub top_left_y: f32,
    pub bottom_right_x: f32,
    pub bottom_right_y: f32,
    pub confidence: f32,
    pub age: Option<f32>,
    pub gender: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FacesResponse {
    pub faces: Vec<FaceBoundingBox>,
}

