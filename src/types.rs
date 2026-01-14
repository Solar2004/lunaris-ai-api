use serde::{Deserialize, Serialize};

/// Represents a single message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,      // "user", "assistant", "system"
    pub content: String,
}

/// Request payload for chat endpoint
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub provider: Option<String>,  // "auto", "groq", "cerebras", etc.
    #[serde(default)]
    pub model: Option<String>,     // Specific model name to requested
    #[serde(default)]
    pub stream: bool,
}

/// Response for chat endpoint (non-streaming)
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub content: String,
    pub model: String,
    pub provider: String,
}

/// Provider statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStats {
    pub name: String,
    pub requests_used: u32,
    pub requests_limit: u32,
    pub percentage_used: f64,
    pub avg_latency_ms: u64,
    pub error_count: u32,
}

/// Rate limit statistics from provider headers
#[derive(Debug, Clone, Default)]
pub struct RateLimitStats {
    pub remaining: Option<u32>,
    pub limit: Option<u32>,
    pub reset: Option<chrono::DateTime<chrono::Utc>>,
}
