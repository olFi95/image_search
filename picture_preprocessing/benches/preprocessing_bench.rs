use criterion::{criterion_group, criterion_main, Criterion};
use image::DynamicImage;
use picture_preprocessing::preprocessor::Preprocessor;
use std::hint::black_box;
use criterion::async_executor::FuturesExecutor;

fn bench_image_preparation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let preprocessor = rt.block_on(async { Preprocessor::new().await });
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
                b.to_async(FuturesExecutor).iter(|| async {
                    let _ = preprocessor.preprocess(black_box(&image)).await;
                });
            },
        );
    }
}



criterion_group!(benches,
    bench_image_preparation,
);
criterion_main!(benches);

