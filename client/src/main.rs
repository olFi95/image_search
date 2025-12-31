pub mod app;
pub mod header;
pub mod image_card;
pub mod image_grid;
mod image_modal;

use crate::app::App;
use leptos::mount::mount_to_body;

fn main() {
    mount_to_body(App);
}
