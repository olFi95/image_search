use crate::image_modal::ImageModal;
use data::ImageReference;
use leptos::prelude::*;
use leptos::*;

#[component]
pub fn ImageCard(image: ImageReference, marked_images: RwSignal<Vec<String>>) -> impl IntoView {
    let (is_open, set_is_open) = signal(false);
    let image_path = image.image_path.clone();
    let image_path_for_click = image_path.clone();
    let image_path_for_checkbox = image_path.clone();
    let checkbox_click = {
        move |_| {
            let mut current = marked_images.get();
            let already_marked = current.iter().any(|m| *m == image_path_for_checkbox);

            if already_marked {
                // Entfernen
                current.retain(|m| *m != image_path);
            } else {
                // Hinzufügen
                current.push(image_path.clone());
            }

            marked_images.set(current);
        }
    };

    view! {
        <div
            style="
                border: 1px solid #ccc;
                border-radius: 8px;
                background-color: #646472;
                height: 300px;
                display: flex;
                flex-direction: column;
                overflow: hidden;            "
        >
            <div style="padding: 0.25rem;">
                <input type="checkbox" on:click=checkbox_click />
            </div>

            <div style="
                flex-grow: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                cursor: pointer;
            ">
                <img
                    src=image_path_for_click.clone()
                    alt="Bild"
                    style="
                        max-width: 100%;
                        max-height: 100%;
                        object-fit: contain;
                        display: block;
                    "
                    on:click=move |_| set_is_open.set(true)
                />
            </div>
        </div>

        <Show when=move || is_open.get() fallback=|| ()>
            <ImageModal
                image_path=image_path_for_click.clone() // ✅ sicher zu benutzen
                on_close=move || set_is_open.set(false)
            />
        </Show>
    }
}
