use crate::age_gender;
use burn::prelude::{Backend, Device};
use burn::Tensor;
use image::DynamicImage;
use rayon::prelude::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use crate::utils::preprocess_clip;

pub struct FaceAgeAndGenderEstimator<B: Backend> {
    pub model: age_gender::Model<B>,
    pub device: Device<B>,
}
impl<B: Backend> FaceAgeAndGenderEstimator<B> {
    pub fn new(model_path: &str, device: Device<B>) -> Self {
        let model = age_gender::Model::from_file(
            model_path,
            &device,
        );
        Self {
            model,
            device,
        }
    }

    pub fn embed(&self, face_images: &[&DynamicImage]) -> Vec<Vec<f32>> {
        if face_images.is_empty() {
            return Vec::new();
        }

        let preprocessed: Vec<_> = face_images
            .par_iter()
            .map(|img| preprocess_clip(img))
            .collect();


        let batch = Tensor::cat(preprocessed, 0);
        let age_and_gender = self.model.forward(batch);
        let data_binding = age_and_gender
            .to_data();
        let data = data_binding
            .as_slice::<f32>()
            .unwrap();

        let batch_size = face_images.len();
        let dim = 2;

        (0..batch_size)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                data[start..end].to_vec()
            })
            .collect()
    }
}
