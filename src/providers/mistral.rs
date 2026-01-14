use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct MistralProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
    stream: bool,
}

#[derive(Serialize)]
struct MistralMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MistralResponse {
    choices: Vec<MistralChoice>,
}

#[derive(Deserialize)]
struct MistralChoice {
    delta: Option<MistralDelta>,
}

#[derive(Deserialize)]
struct MistralDelta {
    content: Option<String>,
}

impl MistralProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }
    
    fn default_model(&self) -> &str {
        "mistral-small-latest"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        60 // Default RPM for Pay-as-you-go/tier 1
    }
    
    fn daily_limit(&self) -> u32 {
        2000 // Approximate daily limit
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
    
    async fn available_models(&self) -> Vec<String> {
        let url = "https://api.mistral.ai/v1/models";
        match self.client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .send()
            .await 
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(data) = json["data"].as_array() {
                        return data.iter()
                            .filter_map(|m| m["id"].as_str())
                            .filter(|id| {
                                !id.contains("embed") && 
                                !id.contains("moderation") && 
                                !id.contains("ocr")
                            })
                            .map(|s| s.to_string())
                            .collect();
                    }
                }
                // Fallback to defaults if parsing fails
                vec![self.default_model().to_string()]
            }
            _ => {
                // Return defaults if API call fails
                vec![
                    "mistral-small-latest".to_string(),
                    "mistral-medium-latest".to_string(),
                    "mistral-large-latest".to_string(),
                    "pixtral-12b-latest".to_string(),
                ]
            }
        }
    }
    
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        let model = model.unwrap_or(self.default_model()).to_string();
        
        let mistral_messages: Vec<MistralMessage> = messages
            .into_iter()
            .map(|m| MistralMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = MistralRequest {
            model,
            messages: mistral_messages,
            stream: true,
        };
        
        // Mistral uses standard Bearer auth and has OpenAI-compatible endpoints
        let response = self.client
            .post("https://api.mistral.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Mistral API error {}: {}", status, error_text));
        }

        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        // Mistral typically returns rate limits in headers
        // x-ratelimit-remaining, x-ratelimit-limit, etc.
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
        
        if let Some(reset) = headers.get("x-ratelimit-reset")
             .and_then(|h| h.to_str().ok())
             .and_then(|s| s.parse::<i64>().ok())
        {
             // Convert unix timestamp to DateTime
             stats.reset = chrono::DateTime::from_timestamp(reset, 0).map(|dt| dt.with_timezone(&chrono::Utc));
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
                                if json_str.trim() == "[DONE]" {
                                    break;
                                }
                                
                                if let Ok(response) = serde_json::from_str::<MistralResponse>(json_str) {
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
