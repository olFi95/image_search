use std::sync::Arc;
use burn::backend::Wgpu;
use burn::prelude::{Device, TensorData};
use burn::Tensor;
use image::DynamicImage;
use crate::clip;
use crate::utils::preprocess_clip;

pub struct ImageEmbedder {
    pub model: Arc<Box<clip::Model<Wgpu>>>,
    pub device: Arc<Box<Device<Wgpu>>>,
}
impl ImageEmbedder {
    pub fn new(model_path: &str, device: Arc<Box<Device<Wgpu>>>) -> Self {
        let model = Box::new(clip::Model::from_file(
            model_path,
            device.as_ref().as_ref(),
        ));
        ImageEmbedder {
            model: Arc::new(model),
            device,
        }
    }
    pub fn embed(&self, image: &DynamicImage) -> Vec<f32> {
        let preprocessed_image = preprocess_clip(&image);
        let embedding = self.model.forward(preprocessed_image);
        let embedding = embedding.reshape([768]);
        let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
        let embedding = embedding / norm;
        embedding.to_data().as_slice::<f32>().unwrap().to_vec()
    }
}
