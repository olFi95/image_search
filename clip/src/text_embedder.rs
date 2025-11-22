use embed_anything::embeddings::embed::{Embedder, EmbeddingResult};
use log::info;

pub struct TextEmbedder {
    embedder: Embedder,
}

impl TextEmbedder {
    pub async fn embed_text_single(&self, text: &str) -> Result<Vec<f32>, anyhow::Error>{
        self.embedder.embed(&[text], None, None)
            .await?
            .first().expect("No embedding result found")
            .to_dense()
    }
}

impl TextEmbedder {
    pub fn new() -> Self {
        let clip_embedder =
            Embedder::from_pretrained_hf("Clip", "openai/clip-vit-large-patch14", None, None, None).expect("Failed to create Embedder");
        Self{embedder: clip_embedder}
    }
}