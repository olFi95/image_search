pub mod face_detector;
mod face_embedder;

extern crate alloc;


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
