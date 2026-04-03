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

/// Request payload for embeddings endpoint
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// Response object for a single embedding
#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String, // "embedding"
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Usage statistics for embeddings
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Response for embeddings endpoint
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String, // "list"
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Metadata for a file uploaded to a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: i64,
    pub filename: String,
    pub purpose: String,
}

/// Hyperparameters for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Hyperparameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
}

/// Request to create a fine-tuning job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningRequest {
    pub model: String,
    pub training_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_files: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<Hyperparameters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

/// Information about a fine-tuning job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningJob {
    pub id: String,
    pub model: String,
    pub status: String, // "queued", "running", "succeeded", "failed", "cancelled"
    pub created_at: i64,
    pub finished_at: Option<i64>,
    pub fine_tuned_model: Option<String>,
    pub error: Option<String>,
}
