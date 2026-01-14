use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct GitHubProvider {
    token: String,
    client: Client,
}

#[derive(Serialize)]
struct GitHubRequest {
    model: String,
    messages: Vec<GitHubMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct GitHubMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GitHubResponse {
    choices: Vec<GitHubChoice>,
}

#[derive(Deserialize)]
struct GitHubChoice {
    delta: Option<GitHubDelta>,
}

#[derive(Deserialize)]
struct GitHubDelta {
    content: Option<String>,
}

impl GitHubProvider {
    pub fn new(token: String) -> Self {
        Self { 
            token,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for GitHubProvider {
    fn name(&self) -> &str {
        "github"
    }
    
    fn default_model(&self) -> &str {
        "gpt-4o"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        15 // GitHub Models (Free) has strict limits, usually around 10-15 RPM
    }
    
    fn daily_limit(&self) -> u32 {
        150 // Adjust based on tier
    }
    
    fn is_configured(&self) -> bool {
        !self.token.is_empty()
    }
    
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        let model = model.unwrap_or(self.default_model()).to_string();
        
        let gh_messages: Vec<GitHubMessage> = messages
            .into_iter()
            .map(|m| GitHubMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = GitHubRequest {
            model,
            messages: gh_messages,
            stream: true,
            temperature: Some(0.7),
            max_tokens: Some(4096),
        };
        
        let token = self.token.clone();
        let client = self.client.clone();
        
        // GitHub Models uses the Azure AI inference endpoint
        let response = client
            .post("https://models.inference.ai.azure.com/chat/completions")
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("GitHub Models API error {}: {}", status, error_text));
        }

        // Extract rate limit headers
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
                                
                                if let Ok(response) = serde_json::from_str::<GitHubResponse>(json_str) {
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
