use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};

pub struct JimmyProvider {
    base_url: String,
}

#[derive(Serialize)]
struct JimmyChatOptions {
    #[serde(rename = "selectedModel")]
    selected_model: String,
    #[serde(rename = "systemPrompt")]
    system_prompt: String,
    #[serde(rename = "topK")]
    top_k: u32,
}

#[derive(Serialize)]
struct JimmyRequest {
    messages: Vec<JimmyMessage>,
    #[serde(rename = "chatOptions")]
    chat_options: JimmyChatOptions,
}

#[derive(Serialize, Deserialize)]
struct JimmyMessage {
    role: String,
    content: String,
}



impl JimmyProvider {
    pub fn new(base_url: String) -> Self {
        let mut url = base_url;
        if url.is_empty() || url.contains("127.0.0.1") || url.contains("localhost") {
             url = "https://chatjimmy.ai".to_string();
        }
        
        Self { 
            base_url: url.trim_end_matches('/').to_string(),
        }
    }
}

#[async_trait]
impl AIProvider for JimmyProvider {
    fn name(&self) -> &str {
        "jimmy"
    }
    
    fn default_model(&self) -> &str {
        "llama3.1-8B"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        60 // Same default as other providers for open source rate limits
    }
    
    fn daily_limit(&self) -> u32 {
        14_400
    }
    
    fn is_configured(&self) -> bool {
        // Jimmy Wrapper is often run locally without an API Key
        true
    }
    
    async fn available_models(&self) -> Vec<String> {
        // Fetch models dynamically from ChatJimmy APIs
        let client = reqwest::Client::new();
        let request = client.get(&format!("{}/api/models", self.base_url))
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");

        match request.send().await
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["data"].as_array() {
                        return models
                            .iter()
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
        
        // Extract system prompt
        let system_prompt = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone())
            .unwrap_or_default();
            
        // Filter out system prompt for upstream messages
        let jimmy_messages: Vec<JimmyMessage> = messages
            .into_iter()
            .filter(|m| m.role != "system")
            .map(|m| JimmyMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = JimmyRequest {
            messages: jimmy_messages,
            chat_options: JimmyChatOptions {
                 selected_model: model,
                 system_prompt,
                 top_k: 8,
            }
        };
        
        tracing::info!("Jimmy: Requesting {}", self.base_url);
        
        let client = reqwest::Client::new();
        let req = client
            .post(&format!("{}/api/chat", self.base_url))
            .header("Content-Type", "application/json")
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .json(&request);
            
        let response = match req.send().await
        {
            Ok(r) => r,
            Err(e) => return Err(e.into()),
        };
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            tracing::warn!("Jimmy API error {}: {}", status, error_text);
            return Err(anyhow::anyhow!("Jimmy API error {}: {}", status, error_text));
        }
        
        let stats = crate::types::RateLimitStats::default();
        tracing::info!("✅ Jimmy request succeeded");
        
        let stream = stream! {
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        
                        // Parse upstream text chunks
                        // Upstream just returns standard text blobs ending with <|stats|>...
                        // which we simply push to the client.
                        for line in text.lines() {
                             if line.contains("<|stats|>") {
                                  // Skip stats rendering to client text stream
                                  let cleaned = line.split("<|stats|>").next().unwrap_or_default();
                                  if !cleaned.is_empty() {
                                       yield Ok(cleaned.to_string());
                                  }
                             } else if !line.is_empty() {
                                  yield Ok(line.to_string() + "\n");
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
