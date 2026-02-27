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

    /// Embed a batch of images. Each image is forwarded individually through the GPU
    /// (ONNX models have fixed batch=1 reshapes), but CPU preprocessing is collected upfront.
    /// Returns one Vec<f32> (length 768) per image, in the same order as the input.
    pub fn embed_batch(&self, images: &[&DynamicImage]) -> Vec<Vec<f32>> {
        if images.is_empty() {
            return Vec::new();
        }

        // Preprocess all images on CPU first
        let preprocessed: Vec<_> = images
            .iter()
            .map(|img| preprocess_clip(img))
            .collect();

        // Forward each through GPU individually and normalize
        preprocessed
            .into_iter()
            .map(|tensor| {
                let embedding = self.model.forward(tensor);
                let embedding = embedding.reshape([768]);
                let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
                let embedding = embedding / norm;
                embedding.to_data().as_slice::<f32>().unwrap().to_vec()
            })
            .collect()
    }
}
