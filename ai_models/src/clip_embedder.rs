use crate::clip::{text, vision};
use crate::utils::preprocess_clip;
use burn::prelude::{Backend, Bool, Device, Int, Tensor, TensorData};
use image::DynamicImage;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Mirrors `burn::data::dataloader::batcher::Batcher` without requiring the `dataset` feature.
pub trait Batcher<B: Backend, I, O>: Send + Sync {
    fn batch(&self, items: Vec<I>, device: &B::Device) -> O;
}

/// A single tokenized item ready to be batched.
pub struct ClipTextItem {
    pub token_ids: Vec<i64>,
    pub mask: Vec<bool>,
}

/// A batched set of token tensors and padding masks.
pub struct ClipTextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

/// Batcher that tokenizes raw strings into CLIP token tensors.
pub struct ClipTextBatcher {
    tokenizer: Arc<Tokenizer>,
    max_len: usize,
    pad_id: i64,
}

impl ClipTextBatcher {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        // CLIP uses a fixed sequence length of 77 tokens
        const MAX_LEN: usize = 77;
        let vocab = tokenizer.get_vocab(true);
        let pad_id = *vocab.get("<|endoftext|>").unwrap_or(&0) as i64;
        Self {
            tokenizer,
            max_len: MAX_LEN,
            pad_id,
        }
    }

    fn tokenize(&self, text: &str) -> ClipTextItem {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .expect("Failed to tokenize text");
        let ids = encoding.get_ids();
        let mut token_ids = Vec::with_capacity(self.max_len);
        let mut mask = Vec::with_capacity(self.max_len);

        for i in 0..self.max_len {
            if let Some(&id) = ids.get(i) {
                token_ids.push(id as i64);
                mask.push(false); // not padded
            } else {
                token_ids.push(self.pad_id);
                mask.push(true); // padded
            }
        }

        ClipTextItem { token_ids, mask }
    }
}

impl<B: Backend> Batcher<B, String, ClipTextBatch<B>> for ClipTextBatcher {
    fn batch(&self, items: Vec<String>, device: &B::Device) -> ClipTextBatch<B> {
        let batch_size = items.len();
        let mut flat_tokens: Vec<i64> = Vec::with_capacity(batch_size * self.max_len);
        let mut flat_mask: Vec<bool> = Vec::with_capacity(batch_size * self.max_len);

        for text in &items {
            let item = self.tokenize(text);
            flat_tokens.extend(item.token_ids);
            flat_mask.extend(item.mask);
        }

        let token_data = TensorData::new(flat_tokens, [batch_size, self.max_len]);
        let mask_data = TensorData::new(flat_mask, [batch_size, self.max_len]);

        ClipTextBatch {
            tokens: Tensor::from_data(token_data, device),
            mask_pad: Tensor::from_data(mask_data, device),
        }
    }
}

pub struct ClipEmbedder<B: Backend> {
    pub vision: vision::Model<B>,
    pub text: text::Model<B>,
    pub device: Device<B>,
    batcher: Arc<ClipTextBatcher>,
}
impl<B: Backend> ClipEmbedder<B> {
    pub fn new(vision_model_path: &str, text_model_path: &str, device: Device<B>) -> Self {
        let vision = vision::Model::from_file(vision_model_path, &device);
        let text = text::Model::from_file(text_model_path, &device);
        let tokenizer = Arc::new(
            Tokenizer::from_pretrained("Xenova/clip-vit-large-patch14", None)
                .expect("Failed to create pretrained tokenizer"),
        );
        let batcher = Arc::new(ClipTextBatcher::new(tokenizer));
        ClipEmbedder {
            vision,
            text,
            device,
            batcher,
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

        let items: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let batch = self.batcher.batch(items, &self.device);

        let embeddings = self.text.forward(batch.tokens);

        let norms = (embeddings.clone() * embeddings.clone())
            .sum_dim(1)
            .sqrt();
        let normalized = embeddings / norms;

        let binding = normalized.to_data();
        let data = binding.as_slice::<f32>().unwrap();
        let batch_size = texts.len();
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

    #[test]
    fn test_embed_text_batch() {
        let device = NdArrayDevice::default();
        let image_embedder = ClipEmbedder::<NdArray>::new(VISION_MODEL_PATH, TEXT_MODEL_PATH, device);

        let texts = ["Hello, world!", "Burn is great!", "CLIP embeddings!"];
        let embeddings = image_embedder.embed_text(&texts);

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 768);
        }
    }
}