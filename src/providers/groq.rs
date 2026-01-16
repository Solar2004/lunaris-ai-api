use async_trait::async_trait;
use async_stream::stream;

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::multipart;

pub struct GroqProvider {
    api_keys: Vec<String>,
    current_key_index: AtomicUsize,
}

#[derive(Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

#[derive(Deserialize)]
struct GroqChoice {
    delta: Option<GroqDelta>,
}

#[derive(Deserialize)]
struct GroqDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
struct GroqTranscriptionResponse {
    text: String,
}

impl GroqProvider {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self { 
            api_keys,
            current_key_index: AtomicUsize::new(0),
        }
    }
    
    fn next_api_key(&self) -> Option<String> {
        if self.api_keys.is_empty() {
            return None;
        }
        let index = self.current_key_index.fetch_add(1, Ordering::Relaxed);
        Some(self.api_keys[index % self.api_keys.len()].clone())
    }
}

#[async_trait]
impl AIProvider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
    }
    
    fn default_model(&self) -> &str {
        "llama-3.3-70b-versatile"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        60
    }
    
    fn daily_limit(&self) -> u32 {
        14_400
    }
    
    fn is_configured(&self) -> bool {
        !self.api_keys.is_empty()
    }

    fn supports_transcription(&self) -> bool {
        true
    }
    
    async fn available_models(&self) -> Vec<String> {
        // Fetch models dynamically from Groq API
        let api_key = match self.api_keys.first() {
            Some(key) => key,
            None => return vec![self.default_model().to_string()],
        };
        
        let client = reqwest::Client::new();
        match client
            .get("https://api.groq.com/openai/v1/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
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
        
        // ... (existing messages conversion)
        let groq_messages: Vec<GroqMessage> = messages
            .into_iter()
            .map(|m| GroqMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = GroqRequest {
            model,
            messages: groq_messages,
            stream: true,
        };
        
        // Try each API key until one works
        let mut last_error = None;
        for attempt in 0..self.api_keys.len() {
            let api_key = match self.next_api_key() {
                Some(key) => key,
                None => return Err(anyhow::anyhow!("No API keys configured")),
            };
            
            tracing::info!("Groq: Trying API key {} (attempt {}/{})", 
                &api_key[..8.min(api_key.len())], attempt + 1, self.api_keys.len());
            
            let client = reqwest::Client::new();
            let response = match client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Groq API key failed: {}", e);
                    last_error = Some(e.into());
                    continue;
                }
            };
            
            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                tracing::warn!("Groq API error {}: {}", status, error_text);
                last_error = Some(anyhow::anyhow!("Groq API error {}: {}", status, error_text));
                continue;
            }
            
            // Extract rate limit headers from Groq
            let mut stats = crate::types::RateLimitStats::default();
            let headers = response.headers();
            
            if let Some(rem) = headers.get("x-ratelimit-remaining-requests")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse().ok()) {
                stats.remaining = Some(rem);
            }
            if let Some(limit) = headers.get("x-ratelimit-limit-requests")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse().ok()) {
                stats.limit = Some(limit);
            }

            // Success! Create stream
            tracing::info!("✅ Groq API key {} succeeded", &api_key[..8.min(api_key.len())]);
            
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
                                    
                                    if let Ok(response) = serde_json::from_str::<GroqResponse>(json_str) {
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
            
            return Ok((Box::pin(stream), Some(stats)));
        }
        
        // All keys failed
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All Groq API keys failed")))
    }

    async fn transcribe_file(
        &self,
        file_bytes: Vec<u8>,
        _content_type: &str, // Groq handles most audio types automatically
    ) -> Result<String, anyhow::Error> {
         // Cycle keys if needed, but for now simple retry logic similar to chat
         let mut last_error = None;
         
         for attempt in 0..self.api_keys.len() {
             let api_key = match self.next_api_key() {
                 Some(key) => key,
                 None => return Err(anyhow::anyhow!("No API keys configured")),
             };
             
             let client = reqwest::Client::new();
             
             // Multipart form
             let form = multipart::Form::new()
                 .part("file", multipart::Part::bytes(file_bytes.clone()).file_name("audio.mp3")) // Filename required by some APIs
                 .text("model", "whisper-large-v3")
                 .text("temperature", "0")
                 .text("response_format", "json");
                 
             let response = match client
                 .post("https://api.groq.com/openai/v1/audio/transcriptions")
                 .header("Authorization", format!("Bearer {}", api_key))
                 .multipart(form)
                 .send()
                 .await
             {
                 Ok(r) => r,
                 Err(e) => {
                     last_error = Some(e.into());
                     continue;
                 }
             };
             
             if !response.status().is_success() {
                 let status = response.status();
                 let error_text = response.text().await.unwrap_or_default();
                 last_error = Some(anyhow::anyhow!("Groq transcription error {}: {}", status, error_text));
                 continue;
             }
             
             let res: GroqTranscriptionResponse = response.json().await?;
             return Ok(res.text);
         }
         
         Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All Groq API keys failed for transcription")))
    }
}
