use crate::clip;
use crate::utils::preprocess_clip;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::Device;
use image::DynamicImage;

pub struct ImageEmbedder {
    pub model: Box<clip::Model<Wgpu>>,
    pub device: Device<Wgpu>,
}
impl ImageEmbedder {
    pub fn new(model_path: &str, device: WgpuDevice) -> Self {
        let model = Box::new(clip::Model::from_file(
            model_path,
            &device,
        ));
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
