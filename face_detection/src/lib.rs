pub mod face_detector;
pub mod face_embedder;
pub mod face_age_and_gender_estimator;

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
pub mod age_gender {
    include!(concat!(
    env!("OUT_DIR"),
    "/age_gender/age_gender.rs"
    ));
}
