use leptos::callback::Callback;
use leptos::html::Div;
use leptos::prelude::*;
use leptos::*;
use web_sys::{MouseEvent, WheelEvent};
#[component]
pub fn ImageModal(image_path: String, on_close: impl Fn() + 'static) -> impl IntoView {
    let (scale, set_scale) = create_signal(1.0_f64);
    let (offset, set_offset) = create_signal((0.0_f64, 0.0_f64));
    let container_ref = create_node_ref::<Div>();
    let is_dragging = create_rw_signal(false);
    let (last_mouse_pos, set_last_mouse_pos) = create_signal((0.0, 0.0));

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

    let on_mouse_up = move |_: MouseEvent| {
        is_dragging.set(false);
    };

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
            on:click=move |_| on_close()
        >
            <div
                node_ref=container_ref
                style="width: 100%; height: 100%; position: relative; overflow: hidden;"
                on:wheel=on_wheel
                on:mousemove=on_mouse_move
                on:mouseup=on_mouse_up
                on:mousedown=on_mouse_down
                on:click:stop_propagation=move |_: MouseEvent| {}
            >
                <img
                    src=image_path
                    draggable="false"
                    style=move || {
                        let (ox, oy) = offset.get();
                        let s = scale.get();
                        format!(
                            "\
                            transform: translate({}px, {}px) scale({});\
                            transform-origin: 0 0;\
                            user-select: none;\
                            pointer-events: none;\
                            position: absolute;\
                            top: 0; left: 0;\
                            max-width: none;\
                            ",
                            ox, oy, s
                        )
                    }
                />
            </div>
        </div>
    }
}
