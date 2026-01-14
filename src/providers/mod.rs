use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;
use crate::types::ChatMessage;

/// Result type for streaming responses
pub type StreamResult = Result<String, anyhow::Error>;

/// Boxed stream for provider responses
pub type ProviderStream = Pin<Box<dyn Stream<Item = StreamResult> + Send>>;

/// Trait that all AI providers must implement
#[async_trait]
pub trait AIProvider: Send + Sync {
    /// Provider name (e.g., "groq", "cerebras")
    fn name(&self) -> &str;
    
    /// Default model for this provider
    fn default_model(&self) -> &str;
    
    /// Rate limit (requests per minute)
    fn rate_limit_rpm(&self) -> u32;
    
    /// Daily request limit
    fn daily_limit(&self) -> u32;
    
    /// Chat completion with streaming
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error>;
    
    /// Check if provider is configured (has API key)
    fn is_configured(&self) -> bool;
    
    /// Get list of available models for this provider (async to allow dynamic fetching)
    /// Get list of available models for this provider (async to allow dynamic fetching)
    async fn available_models(&self) -> Vec<String> {
        // Default implementation returns just the default model
        vec![self.default_model().to_string()]
    }
    
    /// Code completion / Fill-In-the-Middle (FIM) support
    /// Returns a stream of text chunks
    async fn completion_stream(
        &self,
        _prompt: String,
        _suffix: Option<String>,
        _model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        Err(anyhow::anyhow!("Code completion not implemented for this provider"))
    }
}

// Placeholder implementations
mod placeholders;

// Re-export provider implementations
pub mod openrouter;
pub mod groq;
pub mod gemini;
pub mod cerebras;
pub mod codestral;
pub mod github;
pub mod nvidia;
pub mod mistral;
pub mod cohere;
