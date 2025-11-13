use leptos::prelude::*;
use leptos::*;
use data::SearchParams;
#[component]
pub fn Header(
    search_term: ReadSignal<String>,
    search_term_set: WriteSignal<String>,
    on_submit: impl Fn(SearchParams) + 'static + Copy,
    on_scan: impl Fn() + 'static + Copy,
    marked_images: RwSignal::<Vec<String>>,
) -> impl IntoView {

    let on_key_down = move |ev: web_sys::KeyboardEvent| {
        if ev.key() == "Enter" {
            let term = search_term.get();
            if !term.trim().is_empty() {
                on_submit(SearchParams{q: term, referenced_images: marked_images.get().clone()});
            }
        }
    };

    view! {
        <header style="
            height: 60px;
            background-color: #646472;
            color: white;
            padding: 0rem 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            font-size: 18px;       
        ">
            <h1>"Image Search"</h1>
            <button
                on:click=move |_| {
                on_scan()
                }
                style="
                    padding: 0.3rem 0.75rem;
                    font-size: 1rem;
                    border-radius: 4px;
                    border: none;
                    background-color: #4caf50;
                    color: white;
                    cursor: pointer;
                "
            >
                "Scan"
            </button>
            <input
                type="search"
                placeholder="Search..."
                prop:value=search_term
                on:input=move |ev| search_term_set.set(event_target_value(&ev))
                on:keydown=on_key_down
                style="
                    padding: 0.15rem;
                    font-size: 1rem;
                    border-radius: 4px;
                    border: none;
                    width: 400px;
                "
            />
        </header>
    }
}