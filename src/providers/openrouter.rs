use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct OpenRouterProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Deserialize)]
struct OpenRouterChoice {
    delta: Option<OpenRouterDelta>,
}

#[derive(Deserialize)]
struct OpenRouterDelta {
    content: Option<String>,
}

impl OpenRouterProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }
    
    fn default_model(&self) -> &str {
        "meta-llama/llama-3.3-70b-instruct:free"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        20
    }
    
    fn daily_limit(&self) -> u32 {
        50
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
    
    async fn available_models(&self) -> Vec<String> {
        // Fetch free models from OpenRouter Frontend API
        let url = "https://openrouter.ai/api/frontend/models/find?fmt=cards&max_price=0&output_modalities=text";
        
        // We use a new client without auth header for this public endpoint, 
        // or just use the same client (OpenRouter might accept it, but cleaner to just fetch public data)
        let client = reqwest::Client::new();
        
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["data"]["models"].as_array() {
                         let free_models: Vec<String> = models
                            .iter()
                            .filter_map(|m| {
                                // We are looking for the 'endpoint' -> 'model_variant_slug'
                                // The user noted: "allenai/molmo-2-8b:free" is in model_variant_slug
                                if let Some(endpoint) = m.get("endpoint") {
                                    if let Some(slug) = endpoint.get("model_variant_slug").and_then(|s| s.as_str()) {
                                        // Double check pricing if needed, but URL is max_price=0.
                                        // Just return the slug.
                                        return Some(slug.to_string());
                                    }
                                }
                                None
                            })
                            .collect();
                            
                        if !free_models.is_empty() {
                            return free_models;
                        }
                    }
                }
                // Fallback if parsing fails
                vec![self.default_model().to_string()]
            }
            _ => vec![self.default_model().to_string()],
        }
    }
    
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        let model = model.unwrap_or(self.default_model()).to_string();
        
        // Convert ChatMessage to OpenRouterMessage
        let or_messages: Vec<OpenRouterMessage> = messages
            .into_iter()
            .map(|m| OpenRouterMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = OpenRouterRequest {
            model,
            messages: or_messages,
            stream: true,
        };
        
        // Clone api_key to move into async block
        let api_key = self.api_key.clone();
        let client = self.client.clone();
        
        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            // Optional headers for OpenRouter rankings/stats
            .header("HTTP-Referer", "https://github.com/lunaris-ai-api") 
            .header("X-Title", "Lunaris AI API")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("OpenRouter API error {}: {}", status, error_text));
        }

        // Extract rate limit headers from OpenRouter (gateway)
        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        // Check standard headers
        if let Some(rem) = headers.get("x-ratelimit-remaining")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok()) 
        {
            stats.remaining = Some(rem);
        }
        
        if let Some(limit) = headers.get("x-ratelimit-limit")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok())
        {
            stats.limit = Some(limit);
        }
        
        // Create stream
        let stream = stream! {
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        
                        for line in text.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..];
                                if json_str == "[DONE]" {
                                    break;
                                }
                                
                                if let Ok(response) = serde_json::from_str::<OpenRouterResponse>(json_str) {
                                    if let Some(choice) = response.choices.first() {
                                        if let Some(delta) = &choice.delta {
                                            if let Some(content) = &delta.content {
                                                yield Ok(content.clone());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Stream error: {}", e));
                        break;
                    }
                }
            }
        };
        
        Ok((Box::pin(stream), Some(stats)))
    }
}
