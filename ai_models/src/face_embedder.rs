use crate::arcface;
use burn::Tensor;
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use burn::prelude::Device;
use image::DynamicImage;

pub struct FaceEmbedder {
    pub model: Box<arcface::Model<Wgpu>>,
    pub device: Device<Wgpu>,
}

impl FaceEmbedder {
    pub fn new(model_path: &str, device: Device<Wgpu>) -> Self {
        let model = Box::new(arcface::Model::from_file(
            model_path,
            &device,
        ));
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

    pub fn preprocess_arcface(img: &DynamicImage) -> Tensor<Wgpu, 4> {
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

        Tensor::<Wgpu, 4>::from_data(
            burn::tensor::TensorData::new(data, [1, 112, 112, 3]),
            &WgpuDevice::DefaultDevice,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::face_detector::FaceDetector;
    use crate::face_embedder::FaceEmbedder;
    use crate::{arcface, yolo};
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use image::open;
    use std::sync::Arc;

    #[test]
    fn embed_all_faces_of_group_photo() {
        let device = WgpuDevice::DefaultDevice;

        let face_detector = {
            let model: yolo::Model<Wgpu> =
                yolo::Model::from_file("../models/yolo.bpk", &device);
            FaceDetector {
                model: Box::new(model),
                device: device.clone(),
            }
        };
        let face_embedder = {
            let model: arcface::Model<Wgpu> =
                arcface::Model::from_file("../models/arcface_model.bpk", &device);
            FaceEmbedder {
                model: Box::new(model),
                device,
            }
        };

        let image = open("../../test_pictures/pexels-fauxels-3184398.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        let mut embeddings = Vec::new();
        for face in faces {
            embeddings.push(face_embedder.embed(face.face_image));
        }
        assert_eq!(embeddings.len(), 7);
    }
}
