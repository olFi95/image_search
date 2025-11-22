pub mod utils;
pub mod text_embedder;

extern crate alloc;
pub mod clip_vit_large_patch14 {
    include!(concat!(
        env!("OUT_DIR"),
        "/clip_vit_large_patch14/vision_model.rs"
    ));
}
