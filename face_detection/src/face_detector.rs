use std::sync::Arc;
use burn::backend::Wgpu;
use burn::prelude::{Device, TensorData};
use burn::Tensor;
use image::DynamicImage;
use crate::yolo;

pub struct FaceDetector{
    pub model: Arc<yolo::Model<Wgpu>>,
    pub device: Arc<Device<Wgpu>>,
}

impl FaceDetector {
    pub fn new(model_path: &str, device: Arc<Device<Wgpu>>) -> Self {
        let model = yolo::Model::from_file(model_path, device.as_ref());
        FaceDetector {
            model: Arc::new(model),
            device,
        }
    }

    /// Detect all faces in an image. returns a vector of cropped face images.
    pub fn detect(&self, img: &DynamicImage) -> Vec<DynamicImage> {

        let scaled_image = scale_image::<640, 640>(img.clone());
        let input =
            image_to_tensor(&scaled_image.scaled_image, &self.device);

        let output = self.model.forward(input);

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
        }
        let final_boxes = nms(boxes, 0.45);
        let mut faces = vec![];
        for b in &final_boxes {
            let face = img.clone().crop(b.xmin as u32, b.ymin as u32, (b.xmax - b.xmin) as u32, (b.ymax- b.ymin) as u32);
            faces.push(face)
        }

        faces
    }
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

fn nms(mut boxes: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
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


#[cfg(test)]
mod tests {
    use crate::yolo;
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use image::open;
    use crate::face_detector::FaceDetector;

    #[test]
    pub fn test_find_faces_group_photo() {
        let device = WgpuDevice::DefaultDevice;
        let model: yolo::Model<Wgpu> = yolo::Model::from_file("../models/yolo.bpk", &device);
        let face_detector = FaceDetector {
            model: alloc::sync::Arc::new(model),
            device: alloc::sync::Arc::new(device),
        };

        let image = open("test/pexels-fauxels-3184398.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        assert_eq!(faces.len(), 7);
    }

    #[test]
    pub fn test_find_faces_no_faces() {
        let device = WgpuDevice::DefaultDevice;
        let model: yolo::Model<Wgpu> = yolo::Model::from_file("../models/yolo.bpk", &device);

        let face_detector = FaceDetector {
            model: alloc::sync::Arc::new(model),
            device: alloc::sync::Arc::new(device),
        };

        let image = open("test/apples_food_fresh_fruits_kiwis_oranges_royalty_free_images-974148.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        assert_eq!(faces.len(), 0);
    }
    #[test]
    pub fn test_find_faces_statue() {
        let device = WgpuDevice::DefaultDevice;
        let model: yolo::Model<Wgpu> = yolo::Model::from_file("../models/yolo.bpk", &device);

        let face_detector = FaceDetector {
            model: alloc::sync::Arc::new(model),
            device: alloc::sync::Arc::new(device),
        };

        let image = open("test/angel_architecture_art_close_up_daylight_outdoors_rock_sculpture-1043652.jpg").expect("Failed to open image");
        let faces = face_detector.detect(&image);
        assert_eq!(faces.len(), 0);
    }
}
