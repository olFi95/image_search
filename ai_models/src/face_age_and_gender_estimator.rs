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

    /// Estimate age and gender for a batch of face images.
    /// Each image is forwarded individually (ONNX models have fixed batch=1 reshapes),
    /// but CPU preprocessing is collected upfront.
    /// Returns one Vec<f32> (length 2: [age, gender]) per face, in input order.
    pub fn embed_batch(&self, face_images: &[&DynamicImage]) -> Vec<Vec<f32>> {
        if face_images.is_empty() {
            return Vec::new();
        }

        // Preprocess all faces on CPU first
        let preprocessed: Vec<_> = face_images
            .iter()
            .map(|img| preprocess_clip(*img))
            .collect();

        // Forward each through GPU individually
        preprocessed
            .into_iter()
            .map(|tensor| {
                let embedding = self.model.forward(tensor);
                let embedding = embedding.reshape([2]);
                embedding.to_data().as_slice::<f32>().unwrap().to_vec()
            })
            .collect()
    }
}
