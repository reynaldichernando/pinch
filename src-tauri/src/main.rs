// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use enigo::*;
use std::sync::Mutex;
use tauri::State;

#[derive(Default)]
struct AppState {
    enigo: Mutex<Enigo>,
    is_mouse_down: Mutex<bool>,
}

#[tauri::command]
fn mouse_action(x: i32, y: i32, pinch: bool, state: State<AppState>) -> () {
    let mut is_mouse_down = state.is_mouse_down.lock().unwrap();
    let mut enigo = state.enigo.lock().unwrap();
    enigo.mouse_move_to(x, y);
    if pinch {
        if !*is_mouse_down {
            *is_mouse_down = true;
            enigo.mouse_down(MouseButton::Left);
        }
    } else {
        if *is_mouse_down {
            enigo.mouse_up(MouseButton::Left);
        }
        *is_mouse_down = false;
    }
}

fn main() {
    let app_state = AppState {
        enigo: Mutex::new(Enigo::new()),
        is_mouse_down: Mutex::new(false),
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![mouse_action])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
