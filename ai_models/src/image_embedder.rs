use burn::prelude::{Backend, Device};
use image::DynamicImage;
use crate::clip;
use crate::utils::preprocess_clip;

pub struct ImageEmbedder<B: Backend> {
    pub model: clip::Model<B>,
    pub device: Device<B>,
}
impl<B: Backend> ImageEmbedder<B> {
    pub fn new(model_path: &str, device: Device<B>) -> Self {
        let model = clip::Model::from_file(
            model_path,
            &device,
        );
        ImageEmbedder {
            model,
            device,
        }
    }
    pub fn embed(&self, image: &DynamicImage) -> Vec<f32> {
        let preprocessed_image = preprocess_clip(image);
        let embedding = self.model.forward(preprocessed_image);
        let embedding = embedding.reshape([768]);
        let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
        let embedding = embedding / norm;
        embedding.to_data().as_slice::<f32>().unwrap().to_vec()
    }
}
