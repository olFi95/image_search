use burn::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use clip::text_embedder::TextEmbedder;
use clip::utils::image_to_resnet;
use criterion::async_executor::FuturesExecutor;
use criterion::{criterion_group, criterion_main, Criterion};
use image::DynamicImage;
use std::hint::black_box;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

fn bench_text_embedding(c: &mut Criterion) {
    c.bench_function("text_embedding using EmbedAnything for single text prompt", |b| {
        let mut iterator = Arc::new(AtomicUsize::new(0));
        let embedder = TextEmbedder::new();
        b.to_async(FuturesExecutor).iter(|| async {
            let _ = embedder.embed_text_single(&format!("Hello, world! {}", iterator.fetch_add(1, Relaxed))).await.unwrap();
        })
    });
}


fn bench_image_embedding_inference(c: &mut Criterion) {
    let device = WgpuDevice::DefaultDevice;
    let model =
        clip::clip_vit_large_patch14::Model::from_file("../models/vision_model.mpk", &device);
    for batch_size in [1, 5, 10, 20, 50, 100] {
        c.bench_function(format!("vision embedding using burn/wgpu batch of {batch_size}").as_str(), |b| {
            let image_buffer = black_box(vec![0;224*224*3*batch_size]);

            b.to_async(FuturesExecutor).iter(|| async {
                let image_tensor_data = burn::tensor::TensorData::new(
                    image_buffer.clone(),
                    [batch_size, 3, 224, 224],
                );
                let image_tensor = Tensor::<Wgpu, 4>::from_data(image_tensor_data.convert::<f32>(), &device);
                model.forward(image_tensor);
            })
        });
    }
}

fn bench_image_preparation(c: &mut Criterion) {
    for image in [
        DynamicImage::new_rgb8(640, 480),
        DynamicImage::new_rgb8(1280, 720),
        DynamicImage::new_rgb8(1920, 1080),
        DynamicImage::new_rgb8(3840, 2160),
    ] {
        c.bench_function(
            format!(
                "Preprocessing of DynamicImage {}x{} for embedding.",
                image.width(),
                image.height()
            )
            .as_str(),
            |b| {
                b.iter(||{
                    let _ = image_to_resnet(black_box(image.clone()));
                });
            },
        );
    }
}


criterion_group!(benches,
    bench_image_preparation,
    bench_text_embedding,
    bench_image_embedding_inference
);
criterion_main!(benches);

