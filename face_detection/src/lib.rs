pub mod face_detector;

extern crate alloc;

use burn::Tensor;
use image::DynamicImage;
use std::sync::Arc;

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
