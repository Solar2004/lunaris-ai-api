use async_trait::async_trait;
use async_stream::stream;
use std::pin::Pin;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream, StreamResult};
use serde::{Deserialize, Serialize};

pub struct CerebrasProvider {
    api_key: String,
}

#[derive(Serialize)]
struct CerebrasRequest {
    model: String,
    messages: Vec<CerebrasMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct CerebrasMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CerebrasResponse {
    choices: Vec<CerebrasChoice>,
}

#[derive(Deserialize)]
struct CerebrasChoice {
    delta: Option<CerebrasDelta>,
}

#[derive(Deserialize)]
struct CerebrasDelta {
    content: Option<String>,
}

impl CerebrasProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

#[async_trait]
impl AIProvider for CerebrasProvider {
    fn name(&self) -> &str {
        "cerebras"
    }
    
    fn default_model(&self) -> &str {
        "llama-3.3-70b"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        30
    }
    
    fn daily_limit(&self) -> u32 {
        43_200  // 30 RPM × 1440 min
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn available_models(&self) -> Vec<String> {
        let url = "https://api.cerebras.ai/v1/models";
        let client = reqwest::Client::new();
        
        match client.get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await 
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(data) = json["data"].as_array() {
                        return data.iter()
                            .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                            .collect();
                    }
                }
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
        let api_key = self.api_key.clone();
        
        let cerebras_messages: Vec<CerebrasMessage> = messages
            .into_iter()
            .map(|m| CerebrasMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = CerebrasRequest {
            model,
            messages: cerebras_messages,
            stream: true,
        };
        
        let client = reqwest::Client::new();
        let response = client
            .post("https://api.cerebras.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Cerebras API error {}: {}",
                status,
                error_text
            ));
        }

        // Extract rate limit headers from Cerebras
        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        if let Some(rem) = headers.get("x-ratelimit-remaining-requests")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok()) 
        {
            stats.remaining = Some(rem);
        }

        if let Some(limit) = headers.get("x-ratelimit-limit-requests")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok())
        {
            stats.limit = Some(limit);
        }
        
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
                                
                                if let Ok(response) = serde_json::from_str::<CerebrasResponse>(json_str) {
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
