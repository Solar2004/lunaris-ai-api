use lunaris_ai_api::{Config, ChatRequest, TrackingService, RequestRecord, hash_api_key};
use serde::Deserialize;
use lunaris_ai_api::providers::{
    AIProvider,
    groq::GroqProvider,
    cerebras::CerebrasProvider,
    gemini::GeminiProvider,
    openrouter::OpenRouterProvider,
    codestral::CodestralProvider,
    github::GitHubProvider,
    nvidia::NvidiaProvider,
    mistral::MistralProvider,
    cohere::CohereProvider,
};
use lunaris_ai_api::storage::StorageEngine;
use axum::{
    routing::{post, get},
    Router,
    Json,
    extract::{State, Query},
    response::{sse::{Event, Sse}, IntoResponse},
    http::StatusCode,
};
use futures::stream::{Stream, StreamExt};
use std::sync::Arc;
use std::convert::Infallible;
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tracing::{info, error, warn};
use chrono::Utc;

// Application state
struct AppState {
    providers: Vec<Arc<dyn AIProvider>>,
    tracking: Arc<TrackingService>,
    config: Config,
}

#[derive(Deserialize)]
struct CompletionRequest {
    provider: String,
    model: Option<String>,
    prompt: String,
    suffix: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Load configuration
    let config = Config::from_env();
    info!("🚀 Starting API-AI-Service on port {}", config.port);
    
    // Initialize providers
    let mut providers: Vec<Arc<dyn AIProvider>> = Vec::new();
    
    // Groq (Priority 1)
    if !config.groq_api_keys.is_empty() {
        providers.push(Arc::new(GroqProvider::new(config.groq_api_keys.clone())));
        info!("✅ Groq provider initialized with {} API key(s)", config.groq_api_keys.len());
    }
    
    // Cerebras (Priority 1)
    if !config.cerebras_api_keys.is_empty() {
        providers.push(Arc::new(CerebrasProvider::new(config.cerebras_api_keys[0].clone())));
        info!("✅ Cerebras provider initialized");
    }
    
    // OpenRouter (Priority 2)
    if !config.openrouter_api_keys.is_empty() {
        providers.push(Arc::new(OpenRouterProvider::new(config.openrouter_api_keys[0].clone())));
        info!("✅ OpenRouter provider initialized");
    }
    
    // Gemini (Priority 3)
    if !config.gemini_api_keys.is_empty() {
        providers.push(Arc::new(GeminiProvider::new(config.gemini_api_keys[0].clone())));
        info!("✅ Gemini provider initialized");
    }
    
    // Codestral (Priority 2)
    if !config.codestral_api_keys.is_empty() {
        providers.push(Arc::new(CodestralProvider::new(config.codestral_api_keys[0].clone())));
        info!("✅ Codestral provider initialized");
    }
    
    // GitHub Models (Priority 3)
    if !config.github_tokens.is_empty() {
        providers.push(Arc::new(GitHubProvider::new(config.github_tokens[0].clone())));
        info!("✅ GitHub Models provider initialized");
    }
    
    // Nvidia NIM (Priority 2)
    if !config.nvidia_api_keys.is_empty() {
        providers.push(Arc::new(NvidiaProvider::new(config.nvidia_api_keys[0].clone())));
        info!("✅ Nvidia NIM provider initialized");
    }
    
    // Mistral La Plateforme (Priority 3)
    if !config.mistral_api_keys.is_empty() {
        providers.push(Arc::new(MistralProvider::new(config.mistral_api_keys[0].clone())));
        info!("✅ Mistral provider initialized");
    }
    
    // Cohere (Priority 4)
    if !config.cohere_api_keys.is_empty() {
        providers.push(Arc::new(CohereProvider::new(config.cohere_api_keys[0].clone())));
        info!("✅ Cohere provider initialized");
    }
    
    if providers.is_empty() {
        error!("❌ No providers configured! Please set API keys in .env");
        std::process::exit(1);
    }
    
    info!("📊 Total providers initialized: {}", providers.len());
    
    // Initialize tracking service
    let storage = Arc::new(StorageEngine::new("api-ai-service.db"));
    let tracking = Arc::new(TrackingService::new(storage));
    info!("📊 Tracking service initialized");
    
    // Create application state
    let state = Arc::new(AppState {
        providers,
        tracking,
        config: config.clone(),
    });
    
    // Build router
    let app = Router::new()
        .route("/chat", post(chat_handler))
        .route("/completions", post(completion_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/models", get(models_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);
    
    // Start server
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    
    info!("🌐 Server listening on http://{}", addr);
    info!("📡 Endpoints:");
    info!("  POST /chat - Chat completion with streaming");
    info!("  POST /completions - Code completion (FIM) with streaming");
    info!("  GET  /health - Health check");
    info!("  GET  /stats - Provider statistics");
    info!("  GET  /models - Available models per provider");
    
    axum::serve(listener, app).await.unwrap();
}

// Health check endpoint
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let provider_count = state.providers.len();
    Json(serde_json::json!({
        "status": "healthy",
        "providers": provider_count,
    }))
}

// Chat endpoint with streaming and tracking
async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    info!("📥 Received chat request with {} messages", request.messages.len());
    
    if state.providers.is_empty() {
        error!("No providers available");
        return Err((StatusCode::SERVICE_UNAVAILABLE, "No providers available".to_string()));
    }
    
    // Select provider logic
    let mut selected_provider = None;
    
    // Check if user requested a specific provider
    let requested_provider = request.provider.as_deref().unwrap_or("auto");
    
    if requested_provider != "auto" {
        // Find the requested provider
        if let Some(p) = state.providers.iter().find(|p| p.name() == requested_provider) {
            // Check limits for requested provider
            if state.tracking.can_accept_request(p.name(), p.rate_limit_rpm()) 
                && state.tracking.has_daily_quota(p.name(), p.daily_limit()) {
                selected_provider = Some(p);
            } else {
                warn!("⚠️ Requested provider {} is rate limited or out of quota", requested_provider);
                // If specific provider requested but not available, we fail rather than fallback
                // to avoid surprise costs/behavior
            }
        } else {
            warn!("⚠️ Requested provider {} not found or not configured", requested_provider);
        }
    } 
    
    // If no specific provider selected (or auto requested), use fallback logic
    if selected_provider.is_none() && requested_provider == "auto" {
        for provider in &state.providers {
            // Check RPM limit
            if !state.tracking.can_accept_request(provider.name(), provider.rate_limit_rpm()) {
                warn!("⏱️ {} at RPM limit, skipping", provider.name());
                continue;
            }
            
            // Check daily quota
            if !state.tracking.has_daily_quota(provider.name(), provider.daily_limit()) {
                warn!("📊 {} daily quota exceeded, skipping", provider.name());
                continue;
            }
            
            selected_provider = Some(provider);
            break;
        }
    }
    
    let provider = match selected_provider {
        Some(p) => p,
        None => {
            error!("All providers exhausted or rate limited");
            return Err((StatusCode::TOO_MANY_REQUESTS, "All providers exhausted or rate limited".to_string()));
        }
    };
    
    let provider_name = provider.name().to_string();
    info!("🎯 Using provider: {}", provider_name);
    
    // Start timing
    let start_time = Instant::now();

    // Get stream from provider
    let (stream, rate_limits) = match provider.chat_stream(request.messages.clone(), request.model.as_deref()).await {
        Ok(s) => s,
        Err(e) => {
            let error_msg = e.to_string();
            error!("Provider error: {}", error_msg);
            
            // Record failed request
            state.tracking.record_request(RequestRecord {
                provider: provider_name.clone(),
                api_key_hash: "unknown".to_string(),
                timestamp: Utc::now(),
                tokens_input: 0,
                tokens_output: 0,
                latency_ms: start_time.elapsed().as_millis() as u64,
                status: "error".to_string(),
                model: provider.default_model().to_string(),
            });
            
            let status_code = if error_msg.contains("429") || error_msg.to_lowercase().contains("quota") || error_msg.to_lowercase().contains("resource exhausted") {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            
            return Err((status_code, error_msg));
        }
    };

    // Log detailed request info to file
    let current_daily = state.tracking.get_requests_today(&provider_name);
    
    // Use real limits if available, otherwise fallback to config
    let (remaining_requests, total_limit) = if let Some(stats) = &rate_limits {
         let limit = stats.limit.unwrap_or(provider.daily_limit());
         // If we have stats.remaining, that's authoritative. 
         // But our logger takes (usage, limit). 
         // If remaining is 144 and limit is 150, usage is 6.
         if let Some(rem) = stats.remaining {
             (rem, limit)
         } else {
             (
                 limit.saturating_sub(current_daily + 1),
                 limit
             )
         }
    } else {
        (
             provider.daily_limit().saturating_sub(current_daily + 1),
             provider.daily_limit()
        )
    };
    
    // We already calculated remaining, but the logger function takes usage/limit.
    // Let's adjust the logger function logic or pass what we have.
    // Ideally we pass "real remaining" explicitly.
    // For now, let's just pass usage based on what we calculated.
    let calculated_usage = total_limit.saturating_sub(remaining_requests);

    log_request_details(
        &provider_name,
        provider.default_model(),
        &request.messages,
        calculated_usage, // Approx usage if real limit known
        total_limit,
    );
    
    // Track successful request
    let tracking = state.tracking.clone();
    let latency_ms = start_time.elapsed().as_millis() as u64;
    
    tracking.record_request(RequestRecord {
        provider: provider_name.clone(),
        api_key_hash: "primary".to_string(),
        timestamp: Utc::now(),
        tokens_input: 0,  // TODO: count actual tokens
        tokens_output: 0,
        latency_ms,
        status: "success".to_string(),
        model: provider.default_model().to_string(),
    });
    
    // Log current usage
    let rpm_used = tracking.get_requests_this_minute(&provider_name);
    let daily_used = tracking.get_requests_today(&provider_name);
    info!("📊 {} usage: {}/{} RPM, {}/{} daily", 
        provider_name, rpm_used, provider.rate_limit_rpm(),
        daily_used, provider.daily_limit());
    
    // Convert to SSE stream
    let sse_stream = stream.map(|result| {
        match result {
            Ok(content) => Ok(Event::default().data(content)),
            Err(e) => {
                error!("Stream error: {}", e);
                Ok(Event::default().data(format!("Error: {}", e)))
            }
        }
    });
    
    Ok(Sse::new(sse_stream))
}

// Stats endpoint
async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut provider_stats = Vec::new();
    
    for provider in &state.providers {
        let stats = state.tracking.get_provider_stats(
            provider.name(),
            provider.daily_limit()
        );
        provider_stats.push(serde_json::json!({
            "name": stats.name,
            "requests_used": stats.requests_used,
            "requests_limit": stats.requests_limit,
            "percentage_used": format!("{:.2}%", stats.percentage_used),
            "rpm_limit": provider.rate_limit_rpm(),
            "rpm_current": state.tracking.get_requests_this_minute(provider.name()),
            "avg_latency_ms": stats.avg_latency_ms,
            "error_count": stats.error_count,
        }));
    }
    
    Json(serde_json::json!({
        "timestamp": Utc::now().to_rfc3339(),
        "providers": provider_stats,
    }))
}

// Models endpoint - returns available models per provider
#[derive(Deserialize)]
struct ModelsQuery {
    provider: Option<String>,
}

async fn models_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ModelsQuery>,
) -> impl IntoResponse {
    let mut provider_models = Vec::new();
    
    for provider in &state.providers {
        // If filter is present, skip providers that don't match
        if let Some(target) = &params.provider {
             if !provider.name().eq_ignore_ascii_case(target) {
                 continue;
             }
        }

        let models = provider.available_models().await;
        provider_models.push(serde_json::json!({
            "provider": provider.name(),
            "default_model": provider.default_model(),
            "available_models": models,
        }));
    }
    
    Json(serde_json::json!({
        "providers": provider_models,
    }))
}

/// Log detailed request information to a file
fn log_request_details(
    provider: &str,
    model: &str,
    messages: &[lunaris_ai_api::ChatMessage],
    usage_current: u32,
    usage_limit: u32
) {
    let system_prompt = messages.iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .unwrap_or("None");

    let timestamp = Utc::now().format("%Y-%m-%d %H:%M:%S");
    
    let remaining = if usage_limit > 0 {
        format!("{}", usage_limit.saturating_sub(usage_current))
    } else {
        "Unlimited".to_string()
    };

    let log_entry = format!(
        "\n[{}] REQUEST START\nProvider: {} | Model: {}\nUsage: {}/{} | Remaining: {}\nSystem Prompt: {}\n----------------------------------------\n",
        timestamp, provider, model, usage_current, usage_limit, remaining, system_prompt
    );

    // Append to file 'requests.log' in the current directory
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("requests.log") 
    {
        if let Err(e) = file.write_all(log_entry.as_bytes()) {
            error!("Failed to write to request log: {}", e);
        }
    } else {
        error!("Failed to open requests.log for writing");
    }
}

// Code completion endpoint (FIM)
async fn completion_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // Find provider
    let provider = state.providers.iter()
        .find(|p| p.name() == request.provider)
        .ok_or((StatusCode::BAD_REQUEST, format!("Provider '{}' not found", request.provider)))?;

    // Check rate limits
    if !state.tracking.can_accept_request(provider.name(), provider.rate_limit_rpm()) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded".to_string()));
    }
    
    // Start timing
    let start_time = Instant::now();
    let provider_name = provider.name().to_string();

    // Get stream
    let (stream, _rate_limits) = match provider.completion_stream(
        request.prompt,
        request.suffix,
        request.model.as_deref()
    ).await {
        Ok(s) => s,
        Err(e) => {
            let error_msg = e.to_string();
            error!("Completion error: {}", error_msg);
            
            // Record failed request (simplified for completions)
            state.tracking.record_request(RequestRecord {
                provider: provider_name.clone(),
                api_key_hash: "unknown".to_string(),
                timestamp: Utc::now(),
                tokens_input: 0,
                tokens_output: 0,
                latency_ms: start_time.elapsed().as_millis() as u64,
                status: "error".to_string(),
                model: request.model.clone().unwrap_or("default".to_string()),
            });
            
            return Err((StatusCode::INTERNAL_SERVER_ERROR, error_msg));
        }
    };
    
    // Map stream to SSE events
    let stream = stream.map(|result| {
        match result {
            Ok(text) => Ok(Event::default().data(text)),
            Err(e) => {
                error!("Stream error: {}", e);
                Ok(Event::default().event("error").data(e.to_string()))
            }
        }
    });

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}
