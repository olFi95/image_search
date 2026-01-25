pub mod app;
pub mod header;
pub mod image_card;
pub mod image_grid;
mod image_modal;

use crate::app::App;
use leptos::mount::mount_to_body;
use tracing::Level;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn main() {
    console_error_panic_hook::set_once();
    let package_name = env!("CARGO_CRATE_NAME");

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(tracing_web::MakeConsoleWriter)
        .pretty()
        .with_filter(Targets::default()
            .with_default(Level::TRACE)
            .with_target(package_name, Level::TRACE)
        );

    tracing_subscriber::registry().with(fmt_layer).init();
    tracing::debug!("Running {package_name}.");

    mount_to_body(App);
}
