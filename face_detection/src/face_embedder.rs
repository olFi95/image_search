use crate::arcface;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::Device;
use burn::Tensor;
use image::DynamicImage;
use std::sync::Arc;

pub struct FaceEmbedder {
    pub model: Arc<arcface::Model<Wgpu>>,
    pub device: Arc<Device<Wgpu>>,
}

impl FaceEmbedder {
    pub fn new(model_path: &str, device: Arc<Device<Wgpu>>) -> Self {
        let model = arcface::Model::from_file(model_path, device.as_ref());
        FaceEmbedder {
            model: Arc::new(model),
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

    pub fn preprocess_arcface(img: &DynamicImage) -> Tensor<Wgpu, 4> {
        let img = img.resize_exact(112, 112, image::imageops::FilterType::Triangle);
        let rgb = img.to_rgb8();
        let mut data = Vec::with_capacity(1 * 112 * 112 * 3);

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

        Tensor::<Wgpu, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 112, 112, 3]),
            &WgpuDevice::DefaultDevice,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::face_detector::FaceDetector;
    use crate::{arcface, yolo};
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use image::open;
    use crate::face_embedder::FaceEmbedder;

    #[test]
    fn embed_all_faces_of_group_photo() {
        let device = Arc::new(WgpuDevice::DefaultDevice);

        let face_detector = {
            let model: yolo::Model<Wgpu> = yolo::Model::from_file("../models/yolo.bpk", device.as_ref());
            FaceDetector {
            model: Arc::new(model),
            device: device.clone(),
        }};
        let face_embedder = {
            let model: arcface::Model<Wgpu> = arcface::Model::from_file("../models/arcface_model.bpk", device.as_ref());
            FaceEmbedder {
            model: Arc::new(model),
                device: device.clone(),
        }};

        let image = open("test/pexels-fauxels-3184398.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        let mut embeddings = Vec::new();
        for face in faces {
            embeddings.push(face_embedder.embed(face.face_image));
        }
        assert_eq!(embeddings.len(), 7);

    }
}

