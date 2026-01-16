use async_trait::async_trait;
use crate::providers::{AIProvider, ProviderStream, TranscriptionStream};
use crate::types::{ChatMessage, RateLimitStats};
use futures::{SinkExt, StreamExt, stream::Stream};
use std::pin::Pin;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use std::sync::atomic::{AtomicUsize, Ordering};
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct AssemblyAIProvider {
    api_keys: Vec<String>,
    current_key_index: AtomicUsize,
    client: Client,
}

#[derive(Serialize)]
struct TranscriptRequest {
    audio_url: String,
}

#[derive(Deserialize)]
struct UploadResponse {
    upload_url: String,
}

#[derive(Deserialize)]
struct TranscriptResponse {
    id: String,
    status: String,
    text: Option<String>,
    error: Option<String>,
}

impl AssemblyAIProvider {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self { 
            api_keys,
            current_key_index: AtomicUsize::new(0),
            client: Client::new(),
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
impl AIProvider for AssemblyAIProvider {
    fn name(&self) -> &str {
        "assemblyai"
    }
    
    fn default_model(&self) -> &str {
        "assemblyai-default"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        60 // Placeholder
    }
    
    fn daily_limit(&self) -> u32 {
        1000 // Placeholder
    }
    
    fn is_configured(&self) -> bool {
        !self.api_keys.is_empty()
    }
    
    fn supports_transcription(&self) -> bool {
        true
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _model: Option<&str>,
    ) -> Result<(ProviderStream, Option<RateLimitStats>), anyhow::Error> {
        Err(anyhow::anyhow!("AssemblyAI does not support chat"))
    }

    async fn transcribe_file(
        &self,
        file_bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<String, anyhow::Error> {
        let api_key = self.next_api_key().ok_or_else(|| anyhow::anyhow!("No AssemblyAI API key configured"))?;
        
        // 1. Upload file
        let upload_url = "https://api.assemblyai.com/v2/upload";
        let resp = self.client.post(upload_url)
            .header("Authorization", &api_key)
            .body(file_bytes)
            .send()
            .await?;
            
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow::anyhow!("AssemblyAI upload failed: {}", error_text));
        }
        
        let upload_res: UploadResponse = resp.json().await?;
        let audio_url = upload_res.upload_url;
        
        // 2. Start transcription
        let transcript_url = "https://api.assemblyai.com/v2/transcript";
        let resp = self.client.post(transcript_url)
            .header("Authorization", &api_key)
            .json(&TranscriptRequest { audio_url })
            .send()
            .await?;
            
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow::anyhow!("AssemblyAI transcription start failed: {}", error_text));
        }
        
        let start_res: TranscriptResponse = resp.json().await?;
        let transcript_id = start_res.id;
        
        // 3. Poll for completion
        let polling_url = format!("https://api.assemblyai.com/v2/transcript/{}", transcript_id);
        
        loop {
            let resp = self.client.get(&polling_url)
                .header("Authorization", &api_key)
                .send()
                .await?;
                
            let poll_res: TranscriptResponse = resp.json().await?;
            
            match poll_res.status.as_str() {
                "completed" => return Ok(poll_res.text.unwrap_or_default()),
                "error" => return Err(anyhow::anyhow!("Transcription failed: {}", poll_res.error.unwrap_or_default())),
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
            }
        }
    }
    
    async fn transcribe_stream(
        &self,
        mut audio_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, anyhow::Error>> + Send>>,
    ) -> Result<TranscriptionStream, anyhow::Error> {
         let api_key = self.next_api_key().ok_or_else(|| anyhow::anyhow!("No AssemblyAI API key configured"))?;
         
         // WebSocket URL
         let ws_url = url::Url::parse_with_params(
             "wss://api.assemblyai.com/v2/realtime/ws",
             &[("sample_rate", "16000")] // Assuming 16k sample rate for now, or make configurable
         )?;
         
         let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(ws_url.as_str())
            .header("Authorization", api_key)
            .body(())?;

         let (ws_stream, _) = connect_async(request).await?;
         let (mut write, mut read) = ws_stream.split();
         
         // Spawn a task to forward audio to WebSocket
         tokio::spawn(async move {
             while let Some(chunk_res) = audio_stream.next().await {
                 if let Ok(chunk) = chunk_res {
                     // AssemblyAI expects binary messages with audio data
                     // Format: partial 16-bit PCM (s16le)
                     let msg = Message::Binary(chunk);
                     if let Err(e) = write.send(msg).await {
                         tracing::error!("Failed to send audio to AssemblyAI: {}", e);
                         break;
                     }
                 }
             }
             // Send terminate message
             let _ = write.send(Message::Text("{\"terminate_session\": true}".to_string())).await;
         });
         
         // Process responses
         let stream = async_stream::stream! {
             while let Some(msg_res) = read.next().await {
                 match msg_res {
                     Ok(msg) => {
                         if let Message::Text(text) = msg {
                             // Parse JSON response
                             if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                 if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                                     if !text.is_empty() {
                                         yield Ok(text.to_string());
                                     }
                                 }
                             }
                         }
                     }
                     Err(e) => {
                         yield Err(anyhow::anyhow!("WebSocket error: {}", e));
                         break;
                     }
                 }
             }
         };
         
         Ok(Box::pin(stream))
    }
}
