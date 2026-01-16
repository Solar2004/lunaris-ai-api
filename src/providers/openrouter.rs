use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;

pub struct OpenRouterProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Deserialize)]
struct OpenRouterChoice {
    delta: Option<OpenRouterDelta>,
    message: Option<OpenRouterMessage>, // Non-streaming
}

#[derive(Deserialize)]
struct OpenRouterDelta {
    content: Option<String>,
}

#[derive(Serialize)]
struct OpenRouterImageRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    stream: bool,
}

impl OpenRouterProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }
    
    fn default_model(&self) -> &str {
        "meta-llama/llama-3.3-70b-instruct:free"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        20
    }
    
    fn daily_limit(&self) -> u32 {
        50
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn supports_image_generation(&self) -> bool {
        true 
    }
    
    async fn available_models(&self) -> Vec<String> {
        // Fetch free models from OpenRouter Frontend API
        let url = "https://openrouter.ai/api/frontend/models/find?fmt=cards&max_price=0&output_modalities=text";
        
        // We use a new client without auth header for this public endpoint, 
        // or just use the same client (OpenRouter might accept it, but cleaner to just fetch public data)
        let client = reqwest::Client::new();
        
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["data"]["models"].as_array() {
                         let free_models: Vec<String> = models
                            .iter()
                            .filter_map(|m| {
                                // We are looking for the 'endpoint' -> 'model_variant_slug'
                                // The user noted: "allenai/molmo-2-8b:free" is in model_variant_slug
                                if let Some(endpoint) = m.get("endpoint") {
                                    if let Some(slug) = endpoint.get("model_variant_slug").and_then(|s| s.as_str()) {
                                        // Double check pricing if needed, but URL is max_price=0.
                                        // Just return the slug.
                                        return Some(slug.to_string());
                                    }
                                }
                                None
                            })
                            .collect();
                            
                        if !free_models.is_empty() {
                            return free_models;
                        }
                    }
                }
                // Fallback if parsing fails
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
        
        // Convert ChatMessage to OpenRouterMessage
        let or_messages: Vec<OpenRouterMessage> = messages
            .into_iter()
            .map(|m| OpenRouterMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        
        let request = OpenRouterRequest {
            model,
            messages: or_messages,
            stream: true,
        };
        
        // Clone api_key to move into async block
        let api_key = self.api_key.clone();
        let client = self.client.clone();
        
        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            // Optional headers for OpenRouter rankings/stats
            .header("HTTP-Referer", "https://github.com/lunaris-ai-api") 
            .header("X-Title", "Lunaris AI API")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("OpenRouter API error {}: {}", status, error_text));
        }

        // Extract rate limit headers from OpenRouter (gateway)
        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        // Check standard headers
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
        
        // Create stream
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
                                
                                if let Ok(response) = serde_json::from_str::<OpenRouterResponse>(json_str) {
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

    async fn generate_image(
        &self,
        prompt: String,
        model: Option<&str>,
        _size: Option<&str>,
        _quality: Option<&str>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let model = model.unwrap_or("google/gemini-2.5-flash-image-preview").to_string(); // Default to a known working image model on OR
        
        // OpenRouter uses chat completions for images
        let request = OpenRouterImageRequest {
            model: model,
            messages: vec![
                OpenRouterMessage {
                    role: "user".to_string(),
                    content: prompt
                }
            ],
            stream: false,
        };

        let client = self.client.clone();
        let api_key = self.api_key.clone();

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://github.com/lunaris-ai-api")
            .header("X-Title", "Lunaris AI API")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
             let error = response.text().await?;
             return Err(anyhow::anyhow!("OpenRouter Image Generation failed: {}", error));
        }

        let response_text = response.text().await?;
        println!("DEBUG: Raw OpenRouter Image Response: {}", response_text);

        let res_json: serde_json::Value = serde_json::from_str(&response_text)?;
        
        // Parse response for image
        // Usually in message.content as Markdown ![image](url) OR standard OpenAI format?
        // OpenRouter normalizes to OpenAI format often. 
        // If it's pure standard OpenAI image generation it would be data: [{url: ...}].
        // But via chat/completions it's likely a markdown image or base64.
        
        if let Some(choices) = res_json["choices"].as_array() {
            if let Some(choice) = choices.first() {
                if let Some(message) = choice["message"].as_object() {
                    if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                         // Check for markdown image format: ![desc](url)
                         if let Some(start) = content.find("](") {
                             if let Some(end) = content[start..].find(")") {
                                 let url = &content[start+2..start+end];
                                 // Download the image
                                 if url.starts_with("http") {
                                     let img_resp = reqwest::get(url).await?;
                                     let bytes = img_resp.bytes().await?.to_vec();
                                     return Ok(bytes);
                                 } else if url.starts_with("data:image/") {
                                     // Base64 data uri
                                      let comma = url.find(',').unwrap_or(0);
                                      let base64_data = &url[comma+1..];
                                      use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
                                      let bytes = BASE64.decode(base64_data)?;
                                      return Ok(bytes);
                                 }
                             }
                         }
                         // Check for images array in message (some providers do this)
                         if let Some(_images) = message.get("images").and_then(|i| i.as_array()) {
                              // If explicit "images" field exists
                              // This assumes it might be an array of urls or base64
                         }
                    }
                }
            }
        }
        
        // If we failed to find an image in the standard chat response,
        // it means the model refused or used a format we didn't parse.
        // Or maybe OpenRouter didn't generate an image.
        Err(anyhow::anyhow!("No image found in OpenRouter response"))
    }
}
