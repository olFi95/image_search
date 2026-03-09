use crate::clip::{text, vision};
use crate::utils::preprocess_clip;
use burn::prelude::{Backend, Device};
use burn::Tensor;
use image::DynamicImage;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use tokenizers::Tokenizer;

pub struct ClipEmbedder<B: Backend> {
    pub vision: vision::Model<B>,
    pub text: text::Model<B>,
    pub device: Device<B>,
    tokenizer: Tokenizer,
}
impl<B: Backend> ClipEmbedder<B> {
    pub fn new(vision_model_path: &str, text_model_path: &str, device: Device<B>) -> Self {
        let vision = vision::Model::from_file(
            vision_model_path,
            &device,
        );
        let text = text::Model::from_file(
            text_model_path,
            &device,
        );
        ClipEmbedder {
            vision,
            text,
            device,
            tokenizer: Tokenizer::from_pretrained("Xenova/clip-vit-large-patch14", None).expect("Failed to create pretrained tokenizer"),
        }
    }

    pub fn embed_images(&self, images: &[&DynamicImage]) -> Vec<Vec<f32>> {
        if images.is_empty() {
            return Vec::new();
        }

        let preprocessed: Vec<_> = images
            .par_iter()
            .map(|img| preprocess_clip(img))
            .collect();
        let batch = Tensor::cat(preprocessed, 0);

        let embeddings = self.vision.forward(batch);

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
        let dim = data.len()/batch_size;

        (0..batch_size)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                data[start..end].to_vec()
            })
            .collect()
    }
    pub fn embed_text(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        const SEQ_LEN: usize = 77;
        let batch_size = texts.len();

        // Jeden Text tokenisieren, auf SEQ_LEN truncaten/padden
        let mut flat: Vec<i32> = Vec::with_capacity(batch_size * SEQ_LEN);
        for text in texts {
            let encoding = self.tokenizer.encode(*text, true).expect("Failed to tokenize text");
            let ids = encoding.get_ids();
            for i in 0..SEQ_LEN {
                flat.push(*ids.get(i).unwrap_or(&0) as i32);
            }
        }

        let token_data = burn::prelude::TensorData::new(flat, [batch_size, SEQ_LEN]);
        let input_ids: Tensor<B, 2, burn::prelude::Int> =
            Tensor::from_data(token_data, &self.device);

        let embeddings = self.text.forward(input_ids);

        let norms = (embeddings.clone() * embeddings.clone())
            .sum_dim(1)
            .sqrt();
        let normalized = embeddings / norms;

        let binding = normalized.to_data();
        let data = binding.as_slice::<f32>().unwrap();
        let dim = data.len() / batch_size;

        (0..batch_size)
            .map(|i| data[i * dim..(i + 1) * dim].to_vec())
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use crate::clip_embedder::ClipEmbedder;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use image::open;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    const VISION_MODEL_PATH: &str = "../models/vision_model.bpk";
    const TEXT_MODEL_PATH: &str = "../models/text_model.bpk";

    #[test]
    fn test_embed_batch_three_images_shape_and_norm() {
        let device = NdArrayDevice::default();
        let image_embedder = ClipEmbedder::<NdArray>::new(VISION_MODEL_PATH, TEXT_MODEL_PATH, device);

        let img1 = open("../test_pictures/1_1.jpg").unwrap();
        let img2 = open("../test_pictures/3_1.jpg").unwrap();
        let img3 = open("../test_pictures/7_1.jpg").unwrap();

        let embeddings: Vec<Vec<f32>> = image_embedder.embed_images(&[&img1, &img2, &img3]);

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
        let image_embedder = ClipEmbedder::<NdArray>::new(VISION_MODEL_PATH, TEXT_MODEL_PATH, device);
        let img1 = open("../test_pictures/1_1.jpg").unwrap();
        let img2 = open("../test_pictures/0_1.jpg").unwrap();
        let img3 = open("../test_pictures/3_1.jpg").unwrap();

        let embeddings: Vec<Vec<f32>> = image_embedder.embed_images(&[&img1, &img2, &img3]);

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
        let image_embedder = ClipEmbedder::<NdArray>::new(VISION_MODEL_PATH, TEXT_MODEL_PATH, device);

        let embeddings = image_embedder.embed_images(&[]);
        assert!(embeddings.is_empty());
    }
}