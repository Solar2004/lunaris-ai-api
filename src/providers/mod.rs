use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;
use crate::types::ChatMessage;

/// Result type for streaming responses
pub type StreamResult = Result<String, anyhow::Error>;

/// Boxed stream for provider responses
pub type ProviderStream = Pin<Box<dyn Stream<Item = StreamResult> + Send>>;

/// Result type for streaming transcription responses (text chunks)
pub type TranscriptionStreamResult = Result<String, anyhow::Error>;

/// Boxed stream for transcription responses
pub type TranscriptionStream = Pin<Box<dyn Stream<Item = TranscriptionStreamResult> + Send>>;

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

    /// Check if provider supports transcription
    fn supports_transcription(&self) -> bool {
        false
    }

    /// Transcribe a pre-recorded audio file
    /// Returns the full transcript text
    async fn transcribe_file(
        &self,
        _file_bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<String, anyhow::Error> {
        Err(anyhow::anyhow!("Transcription not implemented for this provider"))
    }
    
    /// Transcribe a live audio stream (e.g., WebSocket)
    /// Returns a stream of transcript updates
    async fn transcribe_stream(
        &self,
        _audio_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, anyhow::Error>> + Send>>,
    ) -> Result<TranscriptionStream, anyhow::Error> {
        Err(anyhow::anyhow!("Streaming transcription not implemented for this provider"))
    }

    /// Check if provider supports text-to-speech
    fn supports_speech(&self) -> bool {
        false
    }
    
    /// Generate speech from text (TTS)
    /// Returns raw audio bytes (e.g., MP3 or WAV)
    async fn speech(
        &self,
        _text: String,
        _model: Option<&str>,
        _voice: Option<&str>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        Err(anyhow::anyhow!("Speech generation not implemented for this provider"))
    }

    /// Check if provider supports image generation
    fn supports_image_generation(&self) -> bool {
        false
    }
    
    /// Generate an image from a prompt
    /// Returns raw image bytes (PNG/JPG)
    async fn generate_image(
        &self,
        _prompt: String,
        _model: Option<&str>,
        _size: Option<&str>,
        _quality: Option<&str>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        Err(anyhow::anyhow!("Image generation not implemented for this provider"))
    }
    
    /// Check if provider supports video generation
    fn supports_video_generation(&self) -> bool {
        false
    }
    
    /// Generate a video from a prompt
    /// Returns raw video bytes (MP4) or a URL? 
    /// For now, let's assume raw bytes for simple cases, or we might need to change this to a URL string if files are too large.
    /// Veo returns a resource URL primarily, but we can download it.
    async fn generate_video(
        &self,
        _prompt: String,
        _model: Option<&str>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        Err(anyhow::anyhow!("Video generation not implemented for this provider"))
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
pub mod assemblyai;
pub mod speechmatics;
