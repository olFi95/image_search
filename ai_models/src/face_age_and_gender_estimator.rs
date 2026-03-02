use crate::age_gender;
use burn::prelude::{Backend, Device};
use image::DynamicImage;
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

    pub fn embed(&self, face_image: DynamicImage) -> Vec<f32> {
        let preprocessed_face = preprocess_clip(&face_image);
        let embedding = self.model.forward(preprocessed_face);
        let embedding = embedding.reshape([2]);
        embedding.to_data().as_slice::<f32>().unwrap().to_vec()
    }
}
