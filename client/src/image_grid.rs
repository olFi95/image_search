use crate::image_card::ImageCard;
use data::ImageReference;
use leptos::control_flow::For;
use leptos::prelude::*;
use leptos::*;

#[component]
pub fn ImageGrid(
    images: ReadSignal<Vec<ImageReference>>,
    marked_images: RwSignal<Vec<String>>,
) -> impl IntoView {
    let items = move || images.get();

    view! {
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            "
        >
            <For
                each=items
                key=|image| image.id.clone()
                children=move |image| view! {
                    <ImageCard image=image marked_images=marked_images/>
                }
            />
        </div>
    }
}
