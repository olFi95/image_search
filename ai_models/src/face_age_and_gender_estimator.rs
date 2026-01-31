use crate::age_gender;
use burn::backend::Wgpu;
use burn::prelude::Device;
use image::DynamicImage;
use crate::utils::preprocess_clip;

pub struct FaceAgeAndGenderEstimator {
    pub model: Box<age_gender::Model<Wgpu>>,
    pub device: Device<Wgpu>,
}
impl FaceAgeAndGenderEstimator {
    pub fn new(model_path: &str, device: Device<Wgpu>) -> Self {
        let model = Box::new(age_gender::Model::from_file(
            model_path,
            &device,
        ));
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
