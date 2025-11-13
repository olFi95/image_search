use serde_wasm_bindgen::to_value;
use crate::image_grid::ImageGrid;
use crate::header::Header;
use leptos::prelude::*;
use leptos::ev::KeyboardEvent;
use leptos::view;
use leptos::component;
use leptos::IntoView;
use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use data::SearchResponse;
use data::SearchParams;
use gloo_net::http::Request;
use leptos::logging::error;
use leptos::server_fn::codec::Json;
use serde_json::from_str;
use serde_urlencoded::to_string;
use urlencoding::encode;

#[component]
pub fn App() -> impl IntoView {
    let (search_term, set_search_term) = signal(String::new());
    let (results, set_results) = signal(Vec::new());
    let marked_images = RwSignal::<Vec<String>>::new(vec![]);

    let perform_search = move |params: SearchParams| {
        error!("Params before encode: {:?}", params);

        spawn_local(async move {
            match to_value(&params) {
                Ok(js_value) => {
                    let request = Request::post("/search")
                        .header("Content-Type", "application/json")
                        .body(js_sys::JSON::stringify(&js_value).unwrap().as_string().unwrap())
                        .unwrap();

                    if let Ok(response) = request.send().await {
                        if let Ok(text) = response.text().await {
                            if let Ok(parsed) = from_str::<SearchResponse>(&text) {
                                set_results.set(parsed.images);
                            } else {
                                error!("Failed to parse SearchResponse");
                            }
                        } else {
                            error!("Failed to read response text");
                        }
                    } else {
                        error!("Request failed");
                    }
                }
                Err(e) => {
                    error!("Failed to serialize params: {:?}", e);
                }
            }
        });
    };


    let perform_scan = move || {
        spawn_local(async move {
            let url = format!("/scan");
            match Request::get(&url).send().await {
                Ok(response) => {
                    
                }
                Err(e) => log::error!("Fehler beim Abrufen: {:?}", e),
            }
        });
    };

    view! {
        <div style="display: flex; flex-direction: column; height: 100vh; background-color: #161618;">
            <Header search_term search_term_set=set_search_term on_submit=perform_search on_scan=perform_scan marked_images=marked_images />
            <main style="flex: 1; padding-top: 60px;">
                <div style="padding-top: 1rem;">
                    <ImageGrid images=results marked_images=marked_images/>
                </div>
            </main>
        </div>
    }
}
