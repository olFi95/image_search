pub mod face_age_and_gender_estimator;
pub mod face_detector;
pub mod face_embedder;

extern crate alloc;

#[allow(warnings)]
pub mod arcface {
    include!(concat!(env!("OUT_DIR"), "/arcface/arc.rs"));
}
#[allow(warnings)]pub mod yolo {
    include!(concat!(env!("OUT_DIR"), "/yolo/yolo.rs"));
}
#[allow(warnings)]
pub mod age_gender {
    include!(concat!(env!("OUT_DIR"), "/age_gender/age_gender.rs"));
}
