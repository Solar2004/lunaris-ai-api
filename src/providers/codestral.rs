use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct CodestralProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct CodestralRequest {
    model: String,
    messages: Vec<CodestralMessage>,
    stream: bool,
}

#[derive(Serialize)]
struct CodestralMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CodestralResponse {
    choices: Vec<CodestralChoice>,
}

#[derive(Deserialize)]
struct CodestralChoice {
    delta: Option<CodestralDelta>,
}

#[derive(Deserialize)]
struct CodestralDelta {
    content: Option<String>,
}

impl CodestralProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for CodestralProvider {
    fn name(&self) -> &str {
        "codestral"
    }
    
    fn default_model(&self) -> &str {
        "codestral-latest"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        60 // Standard Mistral/Codestral tier
    }
    
    fn daily_limit(&self) -> u32 {
        2000
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
    
    async fn available_models(&self) -> Vec<String> {
        let url = "https://codestral.mistral.ai/v1/models";
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
                            .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                            .collect();
                    }
                }
                vec![self.default_model().to_string()]
            }
            _ => {
                // Return known Codestral models as fallback
                vec![
                    "codestral-latest".to_string(),
                    "codestral-2405".to_string(),
                    "codestral-mamba-latest".to_string(), 
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
        
        let msgs: Vec<CodestralMessage> = messages
            .into_iter()
            .map(|m| CodestralMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = CodestralRequest {
            model,
            messages: msgs,
            stream: true,
        };
        
        // Use the dedicated Codestral endpoint
        let response = self.client
            .post("https://codestral.mistral.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Codestral API error {}: {}", status, error_text));
        }

        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
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
                                
                                if let Ok(response) = serde_json::from_str::<CodestralResponse>(json_str) {
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

    async fn completion_stream(
        &self,
        prompt: String,
        suffix: Option<String>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        let model = model.unwrap_or(self.default_model()).to_string();
        
        #[derive(Serialize)]
        struct FimRequest {
            model: String,
            prompt: String,
            suffix: Option<String>,
            stream: bool,
            temperature: f32,
        }
        
        let request = FimRequest {
            model,
            prompt,
            suffix,
            stream: true,
            temperature: 0.7,
        };

        let response = self.client
            .post("https://codestral.mistral.ai/v1/fim/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Codestral FIM error {}: {}", status, error_text));
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
                                if json_str.trim() == "[DONE]" { break; }
                                
                                // FIM response structure is same as chat in chunks
                                if let Ok(response) = serde_json::from_str::<CodestralResponse>(json_str) {
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

        Ok((Box::pin(stream), None))
    }
}
