use data::{FaceBoundingBox, FacesRequest, FacesResponse};
use gloo_net::http::Request;
use leptos::ev::keydown;
use leptos::html::Div;
use leptos::logging;
use leptos::prelude::*;
use leptos::*;
use leptos_use::use_event_listener;
use serde_json::from_str;
use tracing::trace;
use wasm_bindgen_futures::spawn_local;
use web_sys::{KeyboardEvent, MouseEvent, WheelEvent};
#[component]
pub fn ImageModal(image_path: String, is_open: RwSignal<bool>) -> impl IntoView {
    let (scale, set_scale) = signal(1.0_f64);
    let (offset, set_offset) = signal((0.0_f64, 0.0_f64));
    let container_ref: NodeRef<Div> = NodeRef::new();
    let is_dragging = RwSignal::new(false);
    let (last_mouse_pos, set_last_mouse_pos) = signal((0.0, 0.0));
    let (faces, set_faces) = signal(Vec::<FaceBoundingBox>::new());
    let (show_faces, set_show_faces) = signal(false);

    // Load faces from API when component mounts
    let image_path_for_faces = image_path.clone();
    Effect::new(move |_| {
        let image_path = image_path_for_faces.clone();
        logging::log!("ImageModal mounted, loading faces for: {}", image_path);
        spawn_local(async move {
            let request_body = FacesRequest {
                image_path: image_path.clone(),
            };
            let body_json = serde_json::to_string(&request_body).unwrap();
            logging::log!("Sending faces request: {}", body_json);
            let result = Request::post("/faces")
                .header("Content-Type", "application/json")
                .body(&body_json)
                .unwrap()
                .send()
                .await;
            match result {
                Ok(response) => {
                    logging::log!("Faces response status: {}", response.status());
                    match response.text().await {
                        Ok(text) => {
                            logging::log!("Faces response body: {}", text);
                            match from_str::<FacesResponse>(&text) {
                                Ok(parsed) => {
                                    logging::log!("Parsed {} faces", parsed.faces.len());
                                    set_faces.set(parsed.faces);
                                }
                                Err(e) => {
                                    logging::error!("Failed to parse FacesResponse: {:?}", e);
                                }
                            }
                        }
                        Err(e) => {
                            logging::error!("Failed to read response text: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    logging::error!("Faces request failed: {:?}", e);
                }
            }
        });
    });

    let on_wheel = move |ev: WheelEvent| {
        ev.prevent_default();
        let delta = ev.delta_y();
        let scale_factor: f64 = if delta > 0.0 { 0.9 } else { 1.1 };

        if let Some(container) = container_ref.get() {
            let rect = container.get_bounding_client_rect();
            let mouse_x = ev.client_x() as f64 - rect.left();
            let mouse_y = ev.client_y() as f64 - rect.top();

            let old_scale = scale.get();
            let new_scale = (old_scale * scale_factor).clamp(0.5, 5.0);

            let (ox, oy) = offset.get();
            let new_ox = (ox - mouse_x) * scale_factor + mouse_x;
            let new_oy = (oy - mouse_y) * scale_factor + mouse_y;

            set_offset.set((new_ox, new_oy));
            set_scale.set(new_scale);
        }
    };

    let on_mouse_down = move |ev: MouseEvent| {
        ev.prevent_default();
        is_dragging.set(true);
        trace!("on_mouse_down");
        set_last_mouse_pos.set((ev.client_x() as f64, ev.client_y() as f64));
    };

    let on_mouse_move = move |ev: MouseEvent| {
        if is_dragging.get() {
            let (last_x, last_y) = last_mouse_pos.get();
            let dx = ev.client_x() as f64 - last_x;
            let dy = ev.client_y() as f64 - last_y;
            let (ox, oy) = offset.get();
            set_offset.set((ox + dx, oy + dy));
            set_last_mouse_pos.set((ev.client_x() as f64, ev.client_y() as f64));
        }
    };

    let on_mouse_up = move |_ev: MouseEvent| {
        is_dragging.set(false);
        trace!("on_mouse_up");
    };

    let on_click_close = move |ev: MouseEvent| {
        if !is_dragging.get() {
            trace!("On_click called while not dragging");
            is_open.set(false);
        } else {
            // never happens
            trace!("On_click called when dragging");
        }
        ev.stop_propagation();
    };

    let on_click_close_inner = move |ev: MouseEvent| {
        trace!("Inner on_click called");
        ev.stop_propagation();
    };

    let _ = use_event_listener(document().body(), keydown, move |evt: KeyboardEvent| {
        if evt.key() == "q" {
            trace!("Closing modal image");
            is_open.set(false);
        }
        if evt.key() == "f" {
            set_show_faces.set(!show_faces.get_untracked());
        }
    });

    view! {
        <div
            style="
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0,0,0,0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            "
            on:click=on_click_close
        >
            <div
                node_ref=container_ref
                style="width: 100%; height: 100%; position: relative; overflow: hidden;"
                on:wheel=on_wheel
                on:mousemove=on_mouse_move
                on:mouseup=on_mouse_up
                on:mousedown=on_mouse_down
                on:click=on_click_close_inner
            >
                <div
                    style=move || {
                        let (ox, oy) = offset.get();
                        let s = scale.get();
                        format!(
                            "\
                            transform: translate({}px, {}px) scale({});\
                            transform-origin: 0 0;\
                            position: absolute;\
                            top: 0; left: 0;\
                            display: inline-block;\
                            ",
                            ox, oy, s
                        )
                    }
                >
                    <img
                        src=image_path
                        draggable="false"
                        style="user-select: none; pointer-events: none; display: block; max-width: none;"
                    />
                    <Show when=move || show_faces.get() && !faces.get().is_empty() fallback=|| ()>
                        {move || {
                            faces.get().into_iter().map(|face| {
                                let left = face.top_left_x as f64 * 100.0;
                                let top = face.top_left_y as f64 * 100.0;
                                let width = (face.bottom_right_x - face.top_left_x) as f64 * 100.0;
                                let height = (face.bottom_right_y - face.top_left_y) as f64 * 100.0;
                                let label = match (face.age, face.gender) {
                                    (Some(age), Some(gender)) => {
                                        let gender_str = if gender > 0.5 { "♂" } else { "♀" };
                                        format!("{} ~{:.0}J", gender_str, age)
                                    }
                                    _ => String::new(),
                                };
                                view! {
                                    <div style=format!(
                                        "\
                                        position: absolute;\
                                        left: {left}%; top: {top}%;\
                                        width: {width}%; height: {height}%;\
                                        border: 2px solid #00ff00;\
                                        box-sizing: border-box;\
                                        pointer-events: none;\
                                        "
                                    )>
                                        {if !label.is_empty() {
                                            Some(view! {
                                                <span style="\
                                                    position: absolute;\
                                                    bottom: 100%;\
                                                    left: 0;\
                                                    background: rgba(0,0,0,0.7);\
                                                    color: #00ff00;\
                                                    font-size: 12px;\
                                                    padding: 1px 4px;\
                                                    white-space: nowrap;\
                                                    pointer-events: none;\
                                                ">{label}</span>
                                            })
                                        } else {
                                            None
                                        }}
                                    </div>
                                }
                            }).collect_view()
                        }}
                    </Show>
                </div>
            </div>
        </div>
    }
}
