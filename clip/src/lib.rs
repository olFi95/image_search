extern crate alloc;

use burn::Tensor;
use burn::backend::Wgpu;
use burn::prelude::Device;
use burn::tensor::TensorData;
use image::DynamicImage;
use std::sync::Arc;

#[allow(warnings)]
pub mod clip_vit_large_patch14 {
    include!(concat!(
        env!("OUT_DIR"),
        "/clip_vit_large_patch14/vision_model.rs"
    ));
}

pub struct ImageEmbedder {
    pub model: Arc<Box<clip_vit_large_patch14::Model<Wgpu>>>,
    pub device: Arc<Box<Device<Wgpu>>>,
}
impl ImageEmbedder {
    pub fn new(model_path: &str, device: Arc<Box<Device<Wgpu>>>) -> Self {
        let model = Box::new(clip_vit_large_patch14::Model::from_file(
            model_path,
            device.as_ref().as_ref(),
        ));
        ImageEmbedder {
            model: Arc::new(model),
            device,
        }
    }
    pub fn embed(&self, image: &DynamicImage) -> Vec<f32> {
        let preprocessed_image = Self::preprocess_clip(&image);
        let embedding = self.model.forward(preprocessed_image);
        let embedding = embedding.reshape([768]);
        let norm = (embedding.clone() * embedding.clone()).sum().sqrt();
        let embedding = embedding / norm;
        embedding.to_data().as_slice::<f32>().unwrap().to_vec()
    }
    fn preprocess_clip(img: &DynamicImage) -> Tensor<Wgpu, 4> {
        let resized = img.resize_exact(224, 224, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();
        let pixels = rgb.as_raw().as_slice(); // &[u8] slice in RGBRGBRGB...

        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        // Output in CHW format: [C][H][W]
        let mut data = vec![0.0f32; 3 * 224 * 224];

        for i in 0..(224 * 224) {
            let r = pixels[i * 3] as f32 / 255.0;
            let g = pixels[i * 3 + 1] as f32 / 255.0;
            let b = pixels[i * 3 + 2] as f32 / 255.0;

            data[i] = (r - mean[0]) / std[0]; // Red channel
            data[224 * 224 + i] = (g - mean[1]) / std[1]; // Green channel
            data[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; // Blue channel
        }
        let image_data = TensorData::new(data, [1, 3, 224, 224]);

        image_data.into()
    }
}
