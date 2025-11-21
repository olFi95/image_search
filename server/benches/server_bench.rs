use burn::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use criterion::async_executor::FuturesExecutor;
use criterion::{criterion_group, criterion_main, Criterion};
use embed_anything::embeddings::embed::Embedder;
use image::open;
use std::env::current_dir;
use std::hint::black_box;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use server::image_prepare_resnet;

fn bench_text_embedding(c: &mut Criterion) {
    c.bench_function("text_embedding using EmbedAnything for single text prompt", |b| {
        let mut iterator = Arc::new(AtomicUsize::new(0));
        let clip_embedder =
            Embedder::from_pretrained_hf("Clip", "openai/clip-vit-large-patch14", None, None, None).expect("Could not create embedder");
        b.to_async(FuturesExecutor).iter(|| async {
            clip_embedder.embed(&[&format!("Hello, world! {}", iterator.fetch_add(1,Relaxed))], None, None).await.unwrap();
            let embedding_result = &clip_embedder.embed(&[&"Hello, world!"], None, None).await.unwrap()[0];
            embedding_result.to_dense().unwrap();
        })
    });
}
fn bench_image_embedding(c: &mut Criterion) {
    println!("{}", current_dir().unwrap().display());
    c.bench_function("vision embedding using burn/wgpu", |b| {
        let device = WgpuDevice::DefaultDevice;
        let model =
            clip::clip_vit_large_patch14::Model::from_file("../models/vision_model.mpk", &device);
        let image_file = open("../testdata/pictures/cat.jpg")
            .expect("Could not open image");
        let image_buffer = black_box(image_prepare_resnet(image_file));

        b.to_async(FuturesExecutor).iter(|| async {
            let image_tensor_data = burn::tensor::TensorData::new(
                image_buffer.clone(),
                [1, 3, 224, 224],
            );
            let image_tensor = Tensor::<Wgpu, 4>::from_data(image_tensor_data.convert::<f32>(), &device);
            model.forward(image_tensor);

        })
    });
}


criterion_group!(benches, bench_image_embedding, bench_text_embedding);
criterion_main!(benches);

