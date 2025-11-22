use embed_anything::embeddings::embed::Embedder;
use image::DynamicImage;
use log::info;



pub fn image_to_resnet(img: DynamicImage) -> Vec<f32> {
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
    data
}
