// Public API exports
pub mod types;
pub mod config;
pub mod providers;
pub mod tracking;
pub mod storage;

// Re-export commonly used types
pub use types::{ChatMessage, ChatRequest, ChatResponse, ProviderStats};
pub use config::Config;
pub use providers::AIProvider;
pub use tracking::{TrackingService, RequestRecord, hash_api_key};

// Legacy OpenRouter client (for backward compatibility)

