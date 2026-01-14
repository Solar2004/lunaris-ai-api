use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct CohereProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct CohereRequest {
    model: String,
    messages: Vec<CohereMessage>,
    stream: bool,
}

#[derive(Serialize)]
struct CohereMessage {
    role: String,
    content: CohereContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum CohereContent {
    Text(String),
}

// Response structs for V2 Stream
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum CohereStreamEvent {
    #[serde(rename = "content-delta")]
    ContentDelta { delta: CohereDelta },
    #[serde(rename = "message-start")]
    MessageStart,
    #[serde(rename = "content-start")]
    ContentStart,
    #[serde(rename = "content-end")]
    ContentEnd,
    #[serde(rename = "message-end")]
    MessageEnd,
    // Catch-all for other events like tool-call-start etc
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize, Debug)]
struct CohereDelta {
    message: Option<CohereDeltaMessage>,
}

#[derive(Deserialize, Debug)]
struct CohereDeltaMessage {
    content: Option<CohereDeltaContent>,
}

#[derive(Deserialize, Debug)]
struct CohereDeltaContent {
    text: String,
}

impl CohereProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for CohereProvider {
    fn name(&self) -> &str {
        "cohere"
    }
    
    fn default_model(&self) -> &str {
        "command-r-plus-08-2024"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        20 // Trial limit 20 RPM, Production much higher
    }
    
    fn daily_limit(&self) -> u32 {
        1000 // Placeholder daily limit
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn available_models(&self) -> Vec<String> {
        // Fetch models from V1 endpoint (works for V2 chat models too usually)
        let url = "https://api.cohere.com/v1/models";
        match self.client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await 
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["models"].as_array() {
                        return models.iter()
                            .filter(|m| {
                                // Filter for chat capable models
                                m["endpoints"].as_array()
                                    .map(|arr| arr.iter().any(|e| e == "chat"))
                                    .unwrap_or(false)
                            })
                            .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                            .collect();
                    }
                }
                vec![self.default_model().to_string()]
            }
            _ => vec![self.default_model().to_string()]
        }
    }
    
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        model: Option<&str>,
    ) -> Result<(ProviderStream, Option<crate::types::RateLimitStats>), anyhow::Error> {
        let model = model.unwrap_or(self.default_model()).to_string();
        
        // Convert to V2 format
        let cohere_messages: Vec<CohereMessage> = messages
            .into_iter()
            .map(|m| CohereMessage {
                role: m.role,
                content: CohereContent::Text(m.content),
            })
            .collect();
        
        let request = CohereRequest {
            model,
            messages: cohere_messages,
            stream: true,
        };
        
        // V2 Chat Endpoint
        let response = self.client
            .post("https://api.cohere.com/v2/chat")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Cohere API error {}: {}", status, error_text));
        }

        // Cohere doesn't typically send rate limit headers in chat responses, 
        // but let's check standard ones just in case.
        let mut stats = crate::types::RateLimitStats::default();
        // (Parsing logic similar to others if headers exist)
        
        let stream = stream! {
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        
                        for line in text.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..]; // Skip "data: "
                                
                                // Parse event
                                if let Ok(event) = serde_json::from_str::<CohereStreamEvent>(json_str) {
                                    match event {
                                        CohereStreamEvent::ContentDelta { delta } => {
                                            if let Some(msg) = delta.message {
                                                 if let Some(content) = msg.content {
                                                     yield Ok(content.text);
                                                 }
                                            }
                                        },
                                        // Ignore other events for now
                                        _ => {}
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
