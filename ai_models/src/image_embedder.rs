use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use burn::tensor::module::embedding;
use burn::Tensor;
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

    pub fn embed(&self, images: &[&DynamicImage]) -> Vec<Vec<f32>> {
        if images.is_empty() {
            return Vec::new();
        }

        let preprocessed: Vec<_> = images
            .par_iter()
            .map(|img| preprocess_clip(img))
            .collect();
        let batch = Tensor::cat(preprocessed, 0);

        let embeddings = self.model.forward(batch);

        let norms = (embeddings.clone() * embeddings.clone())
            .sum_dim(1)
            .sqrt();

        let normalized = embeddings / norms;

        let binding = normalized
            .to_data();
        let data = binding
            .as_slice::<f32>()
            .unwrap();

        let batch_size = images.len();
        let dim = 768;

        (0..batch_size)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                data[start..end].to_vec()
            })
            .collect()
    }}
#[cfg(test)]
mod tests {
    use crate::image_embedder::ImageEmbedder;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use image::open;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    const MODEL_PATH: &'static str = "../models/vision_model.bpk";

    #[test]
    fn test_embed_batch_three_images_shape_and_norm() {
        let device = NdArrayDevice::default();
        let image_embedder = ImageEmbedder::<NdArray>::new(MODEL_PATH, device);

        let img1 = open("../test_pictures/1_1.jpg").unwrap();
        let img2 = open("../test_pictures/3_1.jpg").unwrap();
        let img3 = open("../test_pictures/7_1.jpg").unwrap();

        let embeddings: Vec<Vec<f32>> = image_embedder.embed_batch(&[&img1, &img2, &img3]);

        assert_eq!(embeddings.len(), 3);

        for emb in &embeddings {
            assert_eq!(emb.len(), 768);

            let norm = l2_norm(emb);
            assert!((norm - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_embed_batch_different_images_produce_different_embeddings() {
        let device = NdArrayDevice::default();
        let image_embedder = ImageEmbedder::<NdArray>::new(MODEL_PATH, device);
        let img1 = open("../test_pictures/1_1.jpg").unwrap();
        let img2 = open("../test_pictures/0_1.jpg").unwrap();
        let img3 = open("../test_pictures/3_1.jpg").unwrap();

        let embeddings: Vec<Vec<f32>> = image_embedder.embed_batch(&[&img1, &img2, &img3]);

        assert_eq!(embeddings.len(), 3);

        fn dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        let sim_12 = dot(&embeddings[0], &embeddings[1]);
        let sim_13 = dot(&embeddings[0], &embeddings[2]);

        assert!(sim_12 < 0.99);
        assert!(sim_13 < 0.99);
    }

    #[test]
    fn test_embed_batch_empty_input() {
        let device = NdArrayDevice::default();
        let image_embedder = ImageEmbedder::<NdArray>::new(MODEL_PATH, device);

        let embeddings = image_embedder.embed_batch(&[]);
        assert!(embeddings.is_empty());
    }
}