// Placeholder providers - will implement these next

use async_trait::async_trait;
use crate::providers::{AIProvider, ProviderStream};
use crate::types::ChatMessage;

// Gemini Provider moved to src/providers/gemini.rs

// OpenRouter Provider  
// OpenRouter Provider (Implemented in src/providers/openrouter.rs)

// Codestral Provider moved to src/providers/codestral.rs

// GitHub Models Provider
pub struct GitHubProvider {
    token: String,
}

impl GitHubProvider {
    pub fn new(token: String) -> Self {
        Self { token }
    }
}

#[async_trait]
impl AIProvider for GitHubProvider {
    fn name(&self) -> &str { "github" }
    fn default_model(&self) -> &str { "gpt-4o-mini" }
    fn rate_limit_rpm(&self) -> u32 { 10 }
    fn daily_limit(&self) -> u32 { 100 }
    fn is_configured(&self) -> bool { !self.token.is_empty() }
    
    async fn chat_stream(&self, _messages: Vec<ChatMessage>, _model: Option<&str>) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        Err(anyhow::anyhow!("GitHub provider not yet implemented"))
    }
}

// Nvidia Provider moved to src/providers/nvidia.rs

// Mistral Provider moved to src/providers/mistral.rs 

// Cohere Provider
// Cohere Provider (Implemented in src/providers/cohere.rs)
