use async_trait::async_trait;
use crate::providers::{AIProvider, ProviderStream, TranscriptionStream};
use crate::types::{ChatMessage, RateLimitStats};
use futures::{SinkExt, StreamExt, stream::Stream};
use std::pin::Pin;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

use std::sync::atomic::{AtomicUsize, Ordering};
use reqwest::{Client, multipart};
use serde::{Deserialize, Serialize};

pub struct SpeechmaticsProvider {
    api_kets: Vec<String>,
    current_key_index: AtomicUsize,
    client: Client,
    region: String,
}

#[derive(Serialize)]
struct JobConfig {
    type_: String,
    transcription_config: TranscriptionConfig,
}

#[derive(Serialize)]
struct TranscriptionConfig {
    language: String,
}

#[derive(Deserialize)]
struct JobResponse {
    id: String,
}

#[derive(Deserialize)]
struct JobDetails {
    job: JobInfo,
}

#[derive(Deserialize)]
struct JobInfo {
    status: String,
}

impl SpeechmaticsProvider {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self { 
            api_kets: api_keys,
            current_key_index: AtomicUsize::new(0),
            client: Client::new(),
            region: "eu1".to_string(), // Default region
        }
    }
    
    pub fn with_region(mut self, region: String) -> Self {
        self.region = region;
        self
    }
    
    fn next_api_key(&self) -> Option<String> {
        if self.api_kets.is_empty() {
            return None;
        }
        let index = self.current_key_index.fetch_add(1, Ordering::Relaxed);
        Some(self.api_kets[index % self.api_kets.len()].clone())
    }
    
    fn get_base_url(&self) -> String {
        format!("https://{}.asr.api.speechmatics.com/v2", self.region)
    }
}

#[async_trait]
impl AIProvider for SpeechmaticsProvider {
    fn name(&self) -> &str {
        "speechmatics"
    }
    
    fn default_model(&self) -> &str {
        "speechmatics-default"
    }

    fn rate_limit_rpm(&self) -> u32 {
        60 
    }
    
    fn daily_limit(&self) -> u32 {
        1000 
    }
    
    fn is_configured(&self) -> bool {
        !self.api_kets.is_empty()
    }
    
    fn supports_transcription(&self) -> bool {
        true
    }
    
    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _model: Option<&str>,
    ) -> Result<(ProviderStream, Option<RateLimitStats>), anyhow::Error> {
        Err(anyhow::anyhow!("Speechmatics does not support chat"))
    }

    async fn transcribe_file(
        &self,
        file_bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<String, anyhow::Error> {
        let api_key = self.next_api_key().ok_or_else(|| anyhow::anyhow!("No Speechmatics API key configured"))?;
        let base_url = self.get_base_url();
        
        // 1. Create Job with file
        let form = multipart::Form::new()
            .part("data_file", multipart::Part::bytes(file_bytes).file_name("audio.mp3"))
            .part("config", multipart::Part::text(serde_json::to_string(&JobConfig {
                type_: "transcription".to_string(),
                transcription_config: TranscriptionConfig {
                    language: "en".to_string(), // Default to English for now
                }
            })?));
            
        let resp = self.client.post(format!("{}/jobs/", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await?;
            
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow::anyhow!("Speechmatics job creation failed: {}", error_text));
        }
        
        let job_res: JobResponse = resp.json().await?;
        let job_id = job_res.id;
        
        // 2. Poll for completion
        loop {
            let resp = self.client.get(format!("{}/jobs/{}", base_url, job_id))
                .header("Authorization", format!("Bearer {}", api_key))
                .send()
                .await?;
                
            let details_res: JobDetails = resp.json().await?;
            
            match details_res.job.status.as_str() {
                "done" => break,
                "rejected" | "expired" => return Err(anyhow::anyhow!("Job failed with status: {}", details_res.job.status)),
                _ => {
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    continue;
                }
            }
        }
        
        // 3. Get Transcript
        let resp = self.client.get(format!("{}/jobs/{}/transcript", base_url, job_id))
            .header("Authorization", format!("Bearer {}", api_key))
            .query(&[("format", "txt")])
            .send()
            .await?;
            
        if !resp.status().is_success() {
             let error_text = resp.text().await?;
             return Err(anyhow::anyhow!("Failed to retrieve transcript: {}", error_text));
        }
        
        // Return raw text
        let text = resp.text().await?;
        Ok(text)
    }
    
    async fn transcribe_stream(
        &self,
        mut audio_stream: Pin<Box<dyn Stream<Item = Result<Vec<u8>, anyhow::Error>> + Send>>,
    ) -> Result<TranscriptionStream, anyhow::Error> {
         let api_key = self.next_api_key().ok_or_else(|| anyhow::anyhow!("No Speechmatics API key configured"))?;
         
         let ws_url = format!("wss://{}.asr.api.speechmatics.com/v2/en", self.region); 
         
         let request = tokio_tungstenite::tungstenite::http::Request::builder()
            .uri(ws_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .body(())?;

         let (ws_stream, _) = connect_async(request).await?;
         let (mut write, mut read) = ws_stream.split();
         
         // Send configuration message first
         let start_msg = serde_json::json!({
             "message": "StartRecognition",
             "audio_format": {
                 "type": "raw",
                 "encoding": "pcm_s16le",
                 "sample_rate": 16000
             },
             "transcription_config": {
                 "language": "en"
             }
         });
         
         write.send(Message::Text(start_msg.to_string())).await?;
         
         // Forward audio stream
         tokio::spawn(async move {
             while let Some(chunk_res) = audio_stream.next().await {
                 if let Ok(chunk) = chunk_res {
                     let msg = Message::Binary(chunk);
                     if let Err(e) = write.send(msg).await {
                         tracing::error!("Failed to send audio to Speechmatics: {}", e);
                         break;
                     }
                 }
             }
             // Send End of Stream
             let eos_msg = serde_json::json!({"message": "EndOfStream"});
             let _ = write.send(Message::Text(eos_msg.to_string())).await;
         });
         
         // Process responses
         let stream = async_stream::stream! {
             while let Some(msg_res) = read.next().await {
                 match msg_res {
                     Ok(msg) => {
                         if let Message::Text(text) = msg {
                             if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                                 let msg_type = json["message"].as_str().unwrap_or_default();
                                 
                                 if msg_type == "AddTranscript" {
                                      if let Some(results) = json["results"].as_array() {
                                          for result in results {
                                              if let Some(alternatives) = result["alternatives"].as_array() {
                                                  if let Some(best) = alternatives.first() {
                                                      if let Some(content) = best["content"].as_str() {
                                                          yield Ok(content.to_string() + " ");
                                                      }
                                                  }
                                              }
                                          }
                                      }
                                 } else if msg_type == "EndOfTranscript" {
                                     break;
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
