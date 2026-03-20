use std::{
    sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}},
    time::Duration,
};

use ax_ws::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum as ax_ws; // Fixed conflict with internal naming
use futures_util::{sink::SinkExt, stream::StreamExt};
use rust_evades::{config::GameConfig, game::{Action, GameState}, sensing::ObservationBuilder};
use rust_evades_dqn::{trainer::TrainingProgress, network::Network};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::{cors::Any, cors::CorsLayer, services::ServeDir};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod training_manager;
mod cmd_runner;
mod evaluation_manager;

use training_manager::TrainingManager;
use cmd_runner::{CmdRunner, LogLine};
use evaluation_manager::{EvaluationManager, EvaluationProgress};

struct AppState {
    game_state: RwLock<GameState>,
    last_action: RwLock<Action>,
    running: AtomicBool,
    ai_mode: AtomicBool,
    model: RwLock<Option<Network>>,
    obs_builder: RwLock<ObservationBuilder>,
    tx_broadcast: broadcast::Sender<String>,
    training_manager: TrainingManager,
    evaluation_manager: EvaluationManager,
    cmd_runner: CmdRunner,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", content = "data")]
enum HubMessage {
    Game(GameState),
    Training(TrainingProgress),
    Evaluation(EvaluationProgress),
    Log(LogLine),
    Status(EngineStatus),
}

#[derive(Clone, Serialize)]
struct EngineStatus {
    running: bool,
    ai_mode: bool,
    has_model: bool,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = GameConfig::default();
    let game_state = GameState::new(config.clone(), None);
    let (tx_broadcast, _) = broadcast::channel(100);
    let (training_manager, mut training_rx) = TrainingManager::new();
    let (evaluation_manager, mut evaluation_rx) = EvaluationManager::new();
    let (cmd_runner, mut log_rx) = CmdRunner::new();

    let model_path = std::path::Path::new("best_model.json");
    let model = if model_path.exists() {
        if let Ok(content) = std::fs::read_to_string(model_path) {
            use rust_evades_dqn::model::SavedModel;
            if let Ok(saved_model) = serde_json::from_str::<SavedModel>(&content) {
                tracing::info!("Loaded best_model.json");
                Some(Network::from_layers(saved_model.layers))
            } else {
                tracing::error!("Failed to parse best_model.json");
                None
            }
        } else {
            tracing::error!("Failed to read best_model.json");
            None
        }
    } else {
        tracing::info!("No best_model.json found at startup");
        None
    };

    let state = Arc::new(AppState {
        game_state: RwLock::new(game_state),
        last_action: RwLock::new(Action::Idle),
        running: AtomicBool::new(false),
        ai_mode: AtomicBool::new(false),
        model: RwLock::new(model),
        obs_builder: RwLock::new(ObservationBuilder::default()),
        tx_broadcast: tx_broadcast.clone(),
        training_manager,
        evaluation_manager,
        cmd_runner,
    });

    // Bridge training, evaluation and logs to main broadcast
    let tx_bridge = tx_broadcast.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                Ok(progress) = training_rx.recv() => {
                    let msg = HubMessage::Training(progress);
                    let _ = tx_bridge.send(serde_json::to_string(&msg).unwrap());
                }
                Ok(progress) = evaluation_rx.recv() => {
                    let msg = HubMessage::Evaluation(progress);
                    let _ = tx_bridge.send(serde_json::to_string(&msg).unwrap());
                }
                Ok(log) = log_rx.recv() => {
                    let msg = HubMessage::Log(log);
                    let _ = tx_bridge.send(serde_json::to_string(&msg).unwrap());
                }
            }
        }
    });

    // Spawn game loop
    let state_clone = state.clone();
    tokio::spawn(async move {
        loop {
            let fps = state_clone.game_state.read().unwrap().config.render_fps;
            let tick_duration = Duration::from_millis(1000 / fps.max(1));
            tokio::time::sleep(tick_duration).await;

            let is_running = state_clone.running.load(Ordering::SeqCst);
            let ai_mode = state_clone.ai_mode.load(Ordering::SeqCst);
            // ...

            if is_running {
                let action = {
                    if ai_mode {
                        let mut obs_builder = state_clone.obs_builder.write().unwrap();
                        let model = state_clone.model.read().unwrap();
                        let gs = state_clone.game_state.read().unwrap();
                        
                        if let Some(net) = &*model {
                            let obs = obs_builder.build(&gs);
                            let q_values = net.predict(&obs);
                            let best_action_idx = q_values
                                .iter()
                                .enumerate()
                                .fold((0, f32::NEG_INFINITY), |(best_idx, max_q), (idx, &q)| {
                                    if q > max_q { (idx, q) } else { (best_idx, max_q) }
                                })
                                .0;
                            Action::ALL[best_action_idx]
                        } else {
                            *state_clone.last_action.read().unwrap()
                        }
                    } else {
                        *state_clone.last_action.read().unwrap()
                    }
                };

                let mut gs = state_clone.game_state.write().unwrap();
                gs.step_fixed(action);
                
                if gs.done {
                    let seed = if gs.config.default_seed != 0 {
                        Some(gs.config.default_seed)
                    } else {
                        use rand::RngExt;
                        let mut rng = rand::rng();
                        Some(rng.random::<u64>())
                    };
                    gs.reset(seed);
                    
                    let log = LogLine {
                        stream: "system".to_string(),
                        text: format!("[SEED] Simulation reset with seed: {}", gs.base_seed),
                    };
                    let msg = HubMessage::Log(log);
                    let _ = state_clone.tx_broadcast.send(serde_json::to_string(&msg).unwrap());

                    state_clone.obs_builder.write().unwrap().reset(&gs);
                }
            }

            let gs = state_clone.game_state.read().unwrap();
            let msg = HubMessage::Game(gs.clone());
            let _ = state_clone.tx_broadcast.send(serde_json::to_string(&msg).unwrap());

            // Periodically broadcast status
            let status = {
                EngineStatus {
                    running: state_clone.running.load(Ordering::SeqCst),
                    ai_mode: state_clone.ai_mode.load(Ordering::SeqCst),
                    has_model: state_clone.model.read().unwrap().is_some(),
                }
            };
            let status_msg = HubMessage::Status(status);
            let _ = state_clone.tx_broadcast.send(serde_json::to_string(&status_msg).unwrap());
        }
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/config", get(get_config).post(update_config))
        .route("/api/train/start", post(start_training))
        .route("/api/train/stop", post(stop_training))
        .route("/api/train/status", get(get_training_status))
        .route("/api/train/promote", post(promote_model))
        .route("/api/eval/start", post(start_evaluation))
        .route("/api/eval/stop", post(stop_evaluation))
        .route("/api/eval/status", get(get_evaluation_status))
        .route("/api/models", get(list_models))
        .route("/api/models/load", post(load_model))
        .route("/api/run", post(run_command))
        .nest_service("/", ServeDir::new("dist"))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    tracing::info!("listening on {}", listener.local_addr().unwrap());
    ax_ws::serve(listener, app).await.unwrap();
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    path: String,
}

async fn list_models() -> impl IntoResponse {
    let mut models = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();
    
    let locations = vec![
        (".", "Root"),
        ("training_runs/web_run/", "Checkpoints"),
        ("../", "Parent"),
    ];

    let mut checkpoints = Vec::new();
    let mut other_models = Vec::new();

    for (dir, label) in locations {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "json" {
                        let path = entry.path().to_string_lossy().to_string();
                        if seen_paths.contains(&path) { continue; }
                        
                        let file_name = entry.file_name().to_string_lossy().to_string();
                        // Ignore non-model json files
                        if file_name.contains("package") || file_name.contains("tsconfig") {
                            continue;
                        }

                        seen_paths.insert(path.clone());

                        if file_name.starts_with("checkpoint_ep_") {
                            checkpoints.push((file_name, path, label));
                        } else {
                            let name = format!("{} [{}]", file_name, label);
                            other_models.push(ModelInfo { name, path });
                        }
                    }
                }
            }
        }
    }

    // Identify the latest checkpoint
    checkpoints.sort_by(|a, b| b.0.cmp(&a.0));
    if let Some((name, path, label)) = checkpoints.first() {
        models.push(ModelInfo {
            name: format!("{} (Latest) [{}]", name, label),
            path: path.clone(),
        });
    }

    // Add all other models (Best, External, etc.)
    models.extend(other_models);

    // Sort: best_model first
    models.sort_by(|a, b| {
        let a_is_best = a.name.contains("best_model");
        let b_is_best = b.name.contains("best_model");
        if a_is_best && !b_is_best { return std::cmp::Ordering::Less; }
        if !a_is_best && b_is_best { return std::cmp::Ordering::Greater; }
        a.name.cmp(&b.name)
    });

    Json(models)
}

#[derive(Deserialize)]
struct LoadModelRequest {
    path: String,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    let path = std::path::Path::new(&req.path);
    if !path.exists() {
        return (ax_ws::http::StatusCode::NOT_FOUND, "Model file not found").into_response();
    }

    if let Ok(content) = std::fs::read_to_string(path) {
        use rust_evades_dqn::model::SavedModel;
        if let Ok(saved_model) = serde_json::from_str::<SavedModel>(&content) {
            let mut model = state.model.write().unwrap();
            *model = Some(Network::from_layers(saved_model.layers));
            return "Model reloaded successfully".into_response();
        }
    }
    
    (ax_ws::http::StatusCode::BAD_REQUEST, "Failed to load or parse model").into_response()
}

async fn promote_model(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let source = std::path::Path::new("training_runs/web_run/best_model.json");
    let target = std::path::Path::new("best_model.json");
    
    if source.exists() {
        if let Ok(_) = std::fs::copy(source, target) {
            // Reload the model in AppState
            if let Ok(content) = std::fs::read_to_string(target) {
                use rust_evades_dqn::model::SavedModel;
                if let Ok(saved_model) = serde_json::from_str::<SavedModel>(&content) {
                    let mut model = state.model.write().unwrap();
                    *model = Some(Network::from_layers(saved_model.layers));
                    return "Model promoted and reloaded";
                }
            }
            "Model promoted but failed to reload"
        } else {
            "Failed to copy model"
        }
    } else {
        "Source model not found"
    }
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx_broadcast = state.tx_broadcast.subscribe();

    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx_broadcast.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    let state_clone = state.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            if let Ok(input) = serde_json::from_str::<WebInput>(&text) {
                match input {
                    WebInput::Action { action } => {
                        let mut last_action = state_clone.last_action.write().unwrap();
                        *last_action = action;
                    }
                    WebInput::Control { control } => {
                        match control {
                            WebControl::TogglePlay => {
                                let current = state_clone.running.load(Ordering::SeqCst);
                                state_clone.running.store(!current, Ordering::SeqCst);
                            }
                            WebControl::Reset => {
                                let mut gs = state_clone.game_state.write().unwrap();
                                let seed = if gs.config.default_seed != 0 {
                                    Some(gs.config.default_seed)
                                } else {
                                    use rand::RngExt;
                                    let mut rng = rand::rng();
                                    Some(rng.random::<u64>())
                                };
                                gs.reset(seed);

                                let log = LogLine {
                                    stream: "system".to_string(),
                                    text: format!("[SEED] Manual reset with seed: {}", gs.base_seed),
                                };
                                let msg = HubMessage::Log(log);
                                let _ = state_clone.tx_broadcast.send(serde_json::to_string(&msg).unwrap());

                                state_clone.obs_builder.write().unwrap().reset(&gs);
                            }
                            WebControl::ToggleAI => {
                                let current = state_clone.ai_mode.load(Ordering::SeqCst);
                                state_clone.ai_mode.store(!current, Ordering::SeqCst);
                            }
                        }
                    }
                }
            }
        }
    });

    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };
}

#[derive(Deserialize)]
#[serde(tag = "type", content = "data")]
enum WebInput {
    Action { action: Action },
    Control { control: WebControl },
}

#[derive(Deserialize)]
enum WebControl {
    TogglePlay,
    Reset,
    ToggleAI,
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<GameConfig> {
    Json(state.game_state.read().unwrap().config.clone())
}

async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(new_config): Json<GameConfig>,
) -> impl IntoResponse {
    let mut gs = state.game_state.write().unwrap();
    gs.config = new_config;
    
    let seed = if gs.config.default_seed != 0 {
        Some(gs.config.default_seed)
    } else {
        use rand::RngExt;
        let mut rng = rand::rng();
        Some(rng.random::<u64>())
    };
    
    gs.reset(seed);

    state.obs_builder.write().unwrap().reset(&gs);

    let log = LogLine {
        stream: "system".to_string(),
        text: format!("[CONFIG] Applied changes. Seed: {}", gs.base_seed),
    };
    let msg = HubMessage::Log(log);
    let _ = state.tx_broadcast.send(serde_json::to_string(&msg).unwrap());

    "ok"
}

async fn start_training(
    State(state): State<Arc<AppState>>,
    Json(config): Json<rust_evades_dqn::trainer::TrainingConfig>,
) -> impl IntoResponse {
    let output_dir = std::path::PathBuf::from("training_runs/web_run");
    state.training_manager.start(config, output_dir);
    "started"
}

async fn stop_training(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    state.training_manager.stop();
    "stopped"
}

async fn get_training_status(
    State(state): State<Arc<AppState>>,
) -> Json<bool> {
    Json(state.training_manager.is_running())
}

#[derive(Deserialize)]
struct EvalRequest {
    start_seed: u64,
    num_seeds: usize,
}

async fn start_evaluation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvalRequest>,
) -> impl IntoResponse {
    let model = {
        let model_guard = state.model.read().unwrap();
        model_guard.clone()
    };

    if let Some(net) = model {
        let config = state.game_state.read().unwrap().config.clone();
        state.evaluation_manager.start(net, req.start_seed, req.num_seeds, config);
        "started".into_response()
    } else {
        (ax_ws::http::StatusCode::BAD_REQUEST, "No model loaded").into_response()
    }
}

async fn stop_evaluation(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    state.evaluation_manager.stop();
    "stopped"
}

async fn get_evaluation_status(
    State(state): State<Arc<AppState>>,
) -> Json<bool> {
    Json(state.evaluation_manager.is_running())
}

#[derive(Deserialize)]
struct RunRequest {
    command: String,
}

async fn run_command(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RunRequest>,
) -> impl IntoResponse {
    state.cmd_runner.run(req.command);
    "ok"
}
