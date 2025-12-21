extern crate alloc;

use burn::backend::Wgpu;
use burn::prelude::TensorData;
use burn::Tensor;
use burn::tensor::Device;
use image::DynamicImage;

pub mod arcface {
    include!(concat!(
    env!("OUT_DIR"),
    "/arcface/arc.rs"
    ));
}
pub mod yolo {
    include!(concat!(
    env!("OUT_DIR"),
    "/yolo/yolo.rs"
    ));
}


pub struct FaceDetectionResult {
    pub image: DynamicImage,
    pub boxes: Vec<BoundingBox>,
}

pub struct FaceDetectionEmbeddingResult {
    pub image: DynamicImage,
    pub boxes: Vec<BoundingBox>,
}

pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct BoundingBoxWithVector {
    pub face_box: BoundingBox,
    pub vector: Vec<f32>,
}


pub struct ResizedImage {
    pub scaled_image: DynamicImage,
    pub base_x: u32,
    pub base_y: u32,
}

impl ResizedImage {
    pub fn get_scale_factors(&self) -> (f32, f32) {
        let sx = self.base_x as f32 / self.scaled_image.width() as f32;
        let sy = self.base_y as f32 / self.scaled_image.height() as f32;
        (sx, sy)
    }
}

pub fn scale_image<const HEIGHT: u32, const WIDTH: u32>(
    image: DynamicImage,
) -> ResizedImage {
    let resized = image.resize_exact(
        WIDTH,
        HEIGHT,
        image::imageops::FilterType::Triangle,
    );

    ResizedImage {
        scaled_image: resized,
        base_x: image.width(),
        base_y: image.height(),
    }
}
fn image_to_tensor(
    image: &image::DynamicImage,
    device: &Device<Wgpu>,
) -> Tensor<Wgpu, 4> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut data = Vec::with_capacity((3 * width * height) as usize);

    // NCHW layout
    for c in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                data.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    let tensor_data = TensorData::new(
        data,
        [1, 3, height as usize, width as usize],
    );

    Tensor::from_data(tensor_data, device)
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use image::open;
    use crate::{image_to_tensor, nms, scale_image, yolo, BBox};

    #[test]
    pub fn test_find_faces() {
        let device = WgpuDevice::DefaultDevice;
        let model: yolo::Model<Wgpu> = yolo::Model::from_file("../models/yolo.bpk", &device);

        let image = open("image.png").expect("Failed to open image");
        let scaled_image = scale_image::<640, 640>(image.clone());
        let input =
            image_to_tensor(&scaled_image.scaled_image, &device);

        let output = model.forward(input);
        println!("Output: {:?}", output);

        let data = output
            .to_data();
        let data_slice =
            data.as_slice::<f32>()
            .expect("Tensor is not f32");

        let num_attrs = 5;
        let num_anchors = 8400;

        let base = 0; // batch = 0
        let stride_attr = num_anchors;
        let stride_batch = num_attrs * num_anchors;

        let mut boxes = Vec::new();
        let conf_threshold = 0.5;

        for anchor in 0..num_anchors {



            let offset = base * stride_batch + anchor;

            let x    = data_slice[offset + 0 * stride_attr];
            let y    = data_slice[offset + 1 * stride_attr];
            let w    = data_slice[offset + 2 * stride_attr];
            let h    = data_slice[offset + 3 * stride_attr];
            let conf = data_slice[offset + 4 * stride_attr];


            if conf < conf_threshold {
                continue;
            }

            let (sx, sy) = scaled_image.get_scale_factors();
            let xmin = (x - w / 2.0) * sx;
            let ymin = (y - h / 2.0) * sy;
            let xmax = (x + w / 2.0) * sx;
            let ymax = (y + h / 2.0) * sy;

            boxes.push(BBox {
                xmin,
                ymin,
                xmax,
                ymax,
                score: conf,
            });

            println!("Anchor {}: xmin={}, ymin={}, xmax={}, ymax={}, conf={}", anchor, xmin, ymin, xmax, ymax, conf);

        }
        let final_boxes = nms(boxes, 0.45);
        for b in &final_boxes {
            println!(
                "Face [{:.1}, {:.1}, {:.1}, {:.1}] conf {:.2}",
                b.xmin, b.ymin, b.xmax, b.ymax, b.score
            );
            let face = image.clone().crop(b.xmin as u32, b.ymin as u32, (b.xmax - b.xmin) as u32, (b.ymax- b.ymin) as u32);
            face.save("face.jpg").expect("Failed to save face");
        }
    }
}

fn nms(mut boxes: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
    // Sort by confidence (descending)
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();

    while let Some(best) = boxes.pop() {
        keep.push(best.clone());

        boxes.retain(|b| iou(&best, b) < iou_threshold);
    }

    keep
}

fn iou(a: &BBox, b: &BBox) -> f32 {
    let inter_xmin = a.xmin.max(b.xmin);
    let inter_ymin = a.ymin.max(b.ymin);
    let inter_xmax = a.xmax.min(b.xmax);
    let inter_ymax = a.ymax.min(b.ymax);

    let inter_w = (inter_xmax - inter_xmin).max(0.0);
    let inter_h = (inter_ymax - inter_ymin).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    inter_area / (area_a + area_b - inter_area + 1e-6)
}

#[derive(Clone, Debug)]
struct BBox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    score: f32,
}

// pub fn process_image(img: DynamicImage) -> Result<DetectionResult, anyhow::Error> {
//     // 1. Image -> Mat
//     let mat = dynamic_image_to_mat(&img)?;
//
//     // 2. Faces output
//     let mut faces = Vector::<Rect>::new();
//
//     // 3. Params (Haarcascade)
//     let mut params = CParams::new("haarcascade_frontalface_alt.xml")?;
//
//     // 4. Detect faces
//     let found = face::get_faces(&mat, &mut faces, &mut params)?;
//
//     if !found {
//         return Ok(DetectionResult {
//             image: img,
//             boxes: vec![],
//         });
//     }
//
//     // 5. Convert results
//     let mut boxes = Vec::with_capacity(faces.len());
//
//     for rect in faces {
//         boxes.push(BoxVector {
//             face_box: BoundingBox {
//                 x: rect.x.max(0) as u32,
//                 y: rect.y.max(0) as u32,
//                 width: rect.width as u32,
//                 height: rect.height as u32,
//             },
//             vector: Vec::new(), // später ArcFace
//         });
//     }
//
//     Ok(DetectionResult {
//         image: img,
//         boxes,
//     })
// }
//
// fn dynamic_image_to_mat(img: &DynamicImage) -> Result<Mat, anyhow::Error> {
//     let rgb = img.to_rgb8();
//     let (width, height) = rgb.dimensions();
//
//     let mut mat = Mat::from_slice(rgb.as_raw())?;
//     let mat = mat.reshape(3, height as i32)?;
//
//     let mut bgr = Mat::default();
//     imgproc::cvt_color(&mat, &mut bgr, imgproc::COLOR_RGB2BGR, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
//
//     Ok(bgr)
// }
// pub fn preprocess_arcface(img: DynamicImage) -> Tensor<Wgpu, 4> {
//     let img = img.resize_exact(112, 112, image::imageops::FilterType::Triangle);
//     let rgb = img.to_rgb8();
//
//     let mut data = Vec::with_capacity(1 * 3 * 112 * 112);
//
//     // NCHW!
//     for c in 0..3 {
//         for y in 0..112 {
//             for x in 0..112 {
//                 let v = rgb.get_pixel(x, y)[c] as f32;
//                 data.push((v - 127.5) / 128.0);
//             }
//         }
//     }
//
//     let tensor = Tensor::<Wgpu, 4>::from_data(
//         burn::tensor::TensorData::new(data, [1, 3, 112, 112]),
//         &WgpuDevice::DefaultDevice,
//     );
//
//     tensor
// }