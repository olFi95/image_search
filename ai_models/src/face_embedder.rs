use crate::arcface;
use burn::Tensor;
use burn::prelude::{Backend, Device};
use image::DynamicImage;

pub struct FaceEmbedder<B: Backend> {
    pub model: arcface::Model<B>,
    pub device: Device<B>,
}

impl <B: Backend>FaceEmbedder<B> {
    pub fn new(model_path: &str, device: Device<B>) -> Self {
        let model = arcface::Model::from_file(
            model_path,
            &device,
        );
        FaceEmbedder {
            model,
            device,
        }
    }

    /// Generate face embedding from a cropped face image.
    pub fn embed(&self, face_image: DynamicImage) -> Vec<f32> {
        let preprocessed_face = Self::preprocess_arcface(&face_image);
        let embedding = self.model.forward(preprocessed_face);
        let embedding = embedding.reshape([512]);
        let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
        let embedding = embedding / norm;
        embedding.to_data().as_slice::<f32>().unwrap().to_vec()
    }

    /// Generate face embeddings for a batch of cropped face images.
    /// Each image is forwarded individually (ONNX models have fixed batch=1 reshapes),
    /// but preprocessing is collected upfront for better cache locality.
    /// Returns one Vec<f32> (length 512) per face, in the same order as the input.
    pub fn embed_batch(&self, face_images: &[DynamicImage]) -> Vec<Vec<f32>> {
        if face_images.is_empty() {
            return Vec::new();
        }

        // Preprocess all faces on CPU first
        let preprocessed: Vec<Tensor<B, 4>> = face_images
            .iter()
            .map(|img| Self::preprocess_arcface(img))
            .collect();

        // Forward each through GPU individually and normalize
        preprocessed
            .into_iter()
            .map(|tensor| {
                let embedding = self.model.forward(tensor);
                let embedding = embedding.reshape([512]);
                let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
                let embedding = embedding / norm;
                embedding.to_data().as_slice::<f32>().unwrap().to_vec()
            })
            .collect()
    }

    pub fn preprocess_arcface(img: &DynamicImage) -> Tensor<B, 4> {
        let img = img.resize_exact(112, 112, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();
        let mut data = Vec::with_capacity(112 * 112 * 3);

        // NHWC: N=1
        for y in 0..112 {
            for x in 0..112 {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let v = pixel[c] as f32;
                    data.push((v - 127.5) / 128.0);
                }
            }
        }

        Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 112, 112, 3]),
            &B::Device::default(),
        )
    }

}

#[cfg(test)]
mod tests {
    use crate::face_detector::FaceDetector;
    use crate::face_embedder::FaceEmbedder;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use image::open;

    #[test]
    fn embed_all_faces_of_group_photo() {
        let device = NdArrayDevice::default();

        let face_detector = FaceDetector::<NdArray>::new("../models/yolo.bpk", device);
        let face_embedder = FaceEmbedder::<NdArray>::new("../models/arcface_model.bpk", device);

        let image = open("../test_pictures/7_1.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        let mut embeddings = Vec::new();
        for face in faces {
            embeddings.push(face_embedder.embed(face.face_image));
        }
        assert_eq!(embeddings.len(), 7);
    }
}
