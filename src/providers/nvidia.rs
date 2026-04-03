use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct NvidiaProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct NvidiaRequest {
    model: String,
    messages: Vec<NvidiaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct NvidiaMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct NvidiaResponse {
    choices: Vec<NvidiaChoice>,
}

#[derive(Deserialize)]
struct NvidiaChoice {
    delta: Option<NvidiaDelta>,
}

#[derive(Deserialize)]
struct NvidiaDelta {
    content: Option<String>,
}

impl NvidiaProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for NvidiaProvider {
    fn name(&self) -> &str {
        "nvidia"
    }
    
    // Using Llama 3.3 70B as a solid default from Nvidia NIM
    fn default_model(&self) -> &str {
        "meta/llama-3.3-70b-instruct"
    }
    
    // Nvidia trial has rate limits (approx 40 RPM in some sources, but varies)
    fn rate_limit_rpm(&self) -> u32 {
        40 
    }
    
    fn daily_limit(&self) -> u32 {
        1000 // Approximate daily limit for trial
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }
    
    async fn available_models(&self) -> Vec<String> {
        let url = "https://integrate.api.nvidia.com/v1/models";
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
                // Fallback to strict defaults if parsing fails
                vec![self.default_model().to_string()]
            }
            _ => {
                // Return defaults if API call fails
                vec![
                    "meta/llama-3.3-70b-instruct".to_string(),
                    "meta/llama-3.1-405b-instruct".to_string(),
                    "mistralai/mixtral-8x22b-instruct-v0.1".to_string(),
                    "microsoft/phi-3-medium-4k-instruct".to_string(),
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
        
        let nvidia_messages: Vec<NvidiaMessage> = messages
            .into_iter()
            .map(|m| NvidiaMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = NvidiaRequest {
            model,
            messages: nvidia_messages,
            stream: true,
            max_tokens: Some(1024),
        };
        
        // Base URL for Nvidia Hosted NIMs
        let url = "https://integrate.api.nvidia.com/v1/chat/completions";

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            
            let possible_solution = match status.as_u16() {
                401 => "Solution: Check if your API Key is correct in .env file.",
                404 => "Solution: Check if the model name is correct, or if the Nvidia NIM endpoint is reachable. Some models require specific permissions.",
                429 => "Solution: You have hit the rate limit. Please wait before sending more requests.",
                500..=599 => "Solution: Nvidia service is experiencing internal issues. Try again later.",
                _ => "Solution: Check the error message details.",
            };

            return Err(anyhow::anyhow!(
                "Nvidia API error {} - {}\nAPI Key used: {}\n{}", 
                status, 
                error_text,
                self.api_key,
                possible_solution
            ));
        }

        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        // Nvidia NIM (standard OpenAI compatible) often doesn't send x-ratelimit headers 
        // in success responses (sometimes only on failure or depends on the specific NIM).
        // But we check just in case they added it.
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
                                
                                if let Ok(response) = serde_json::from_str::<NvidiaResponse>(json_str) {
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
