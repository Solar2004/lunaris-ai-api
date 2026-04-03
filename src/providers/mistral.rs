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

#[derive(Serialize)]
struct MistralEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
    model: String,
    usage: MistralUsage,
}

#[derive(Deserialize)]
struct MistralEmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct MistralUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize)]
struct MistralFineTuningListResponse {
    data: Vec<crate::types::FinetuningJob>,
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
    async fn embeddings(
        &self,
        model: &str,
        input: Vec<String>,
    ) -> Result<crate::types::EmbeddingResponse, anyhow::Error> {
        let request = MistralEmbeddingRequest {
            model: model.to_string(),
            input,
        };

        let response = self.client
            .post("https://api.mistral.ai/v1/embeddings")
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

        let mistral_response = response.json::<MistralEmbeddingResponse>().await?;

        Ok(crate::types::EmbeddingResponse {
            object: "list".to_string(),
            data: mistral_response.data.into_iter().map(|d| crate::types::EmbeddingObject {
                object: d.object,
                embedding: d.embedding,
                index: d.index,
            }).collect(),
            model: mistral_response.model,
            usage: crate::types::EmbeddingUsage {
                prompt_tokens: mistral_response.usage.prompt_tokens,
                total_tokens: mistral_response.usage.total_tokens,
            },
        })
    }

    async fn upload_file(
        &self,
        data: Vec<u8>,
        filename: &str,
    ) -> Result<crate::types::FileMetadata, anyhow::Error> {
        use reqwest::multipart;

        let part = multipart::Part::bytes(data)
            .file_name(filename.to_string())
            .mime_str("application/octet-stream")?;

        let form = multipart::Form::new()
            .part("file", part)
            .text("purpose", "fine-tune");

        let response = self.client
            .post("https://api.mistral.ai/v1/files")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Mistral File API error {}: {}", status, error_text));
        }

        Ok(response.json::<crate::types::FileMetadata>().await?)
    }

    async fn create_finetuning_job(
        &self,
        request: crate::types::FinetuningRequest,
    ) -> Result<crate::types::FinetuningJob, anyhow::Error> {
        let response = self.client
            .post("https://api.mistral.ai/v1/fine_tuning/jobs")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Mistral Fine-tuning API error {}: {}", status, error_text));
        }

        Ok(response.json::<crate::types::FinetuningJob>().await?)
    }

    async fn get_finetuning_job(
        &self,
        job_id: &str,
    ) -> Result<crate::types::FinetuningJob, anyhow::Error> {
        let url = format!("https://api.mistral.ai/v1/fine_tuning/jobs/{}", job_id);
        let response = self.client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Mistral Fine-tuning API error {}: {}", status, error_text));
        }

        Ok(response.json::<crate::types::FinetuningJob>().await?)
    }

    async fn list_finetuning_jobs(
        &self,
    ) -> Result<Vec<crate::types::FinetuningJob>, anyhow::Error> {
        let response = self.client
            .get("https://api.mistral.ai/v1/fine_tuning/jobs")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Mistral Fine-tuning API error {}: {}", status, error_text));
        }

        let list_resp = response.json::<MistralFineTuningListResponse>().await?;
        Ok(list_resp.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_embedding_request_serialization() {
        let req = MistralEmbeddingRequest {
            model: "mistral-embed".to_string(),
            input: vec!["hello".to_string()],
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"mistral-embed\""));
        assert!(json.contains("\"input\":[\"hello\"]"));
    }

    #[test]
    fn test_file_metadata_deserialization() {
        let json = r#"{
            "id": "file-123",
            "object": "file",
            "bytes": 100,
            "created_at": 123456,
            "filename": "test.jsonl",
            "purpose": "fine-tune"
        }"#;
        let meta: FileMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.id, "file-123");
        assert_eq!(meta.filename, "test.jsonl");
    }

    #[test]
    fn test_finetuning_job_deserialization() {
        let json = r#"{
            "id": "job-123",
            "model": "mistral-small-latest",
            "status": "succeeded",
            "created_at": 123456,
            "finished_at": 123457,
            "fine_tuned_model": "ft:model:suffix"
        }"#;
        let job: FinetuningJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.id, "job-123");
        assert_eq!(job.status, "succeeded");
        assert_eq!(job.fine_tuned_model.unwrap(), "ft:model:suffix");
    }
}
