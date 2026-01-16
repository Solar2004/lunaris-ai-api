use async_trait::async_trait;
use async_stream::stream;
use crate::types::ChatMessage;
use crate::providers::{AIProvider, ProviderStream};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

pub struct GeminiProvider {
    api_key: String,
    client: Client,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}



#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_modalities: Option<Vec<String>>, 
    #[serde(skip_serializing_if = "Option::is_none")]
    speech_config: Option<GeminiSpeechConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiSpeechConfig {
     #[serde(skip_serializing_if = "Option::is_none")]
     multi_speaker_voice_config: Option<GeminiMultiSpeakerConfig>,
     #[serde(skip_serializing_if = "Option::is_none")]
     voice_config: Option<GeminiVoiceConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiMultiSpeakerConfig {
    speaker_voice_configs: Vec<GeminiSpeakerVoiceConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiSpeakerVoiceConfig {
    speaker: String,
    voice_config: GeminiVoiceConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiVoiceConfig {
    prebuilt_voice_config: GeminiPrebuiltVoiceConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPrebuiltVoiceConfig {
    voice_name: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData { inline_data: GeminiInlineData },
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiResponseContent>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseContent {
    parts: Option<Vec<GeminiResponsePart>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponsePart {
    text: Option<String>,
    inline_data: Option<GeminiInlineData>, // For audio response
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl AIProvider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }
    
    fn default_model(&self) -> &str {
        "gemini-1.5-flash"
    }
    
    fn rate_limit_rpm(&self) -> u32 {
        15 // Free tier
    }
    
    fn daily_limit(&self) -> u32 {
        1500 // 1500 request per day free tier approx
    }
    
    fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn supports_transcription(&self) -> bool {
        true
    }
    
    fn supports_speech(&self) -> bool {
        true
    }

    fn supports_image_generation(&self) -> bool {
        true
    }
    
    async fn available_models(&self) -> Vec<String> {
        // Fetch models dynamically from Gemini API
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models?key={}", self.api_key);
        
        let client = reqwest::Client::new();
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(models) = json["models"].as_array() {
                        return models
                            .iter()
                            .filter(|m| {
                                // Filter for models that support 'generateContent'
                                m["supportedGenerationMethods"].as_array()
                                    .map(|methods| methods.iter().any(|method| method == "generateContent"))
                                    .unwrap_or(false)
                            })
                            .filter_map(|m| {
                                // Gemini returns model names like "models/gemini-1.5-pro"
                                m["name"].as_str().and_then(|name| {
                                    name.strip_prefix("models/").map(|s| s.to_string())
                                })
                            })
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
        
        let gemini_contents: Vec<GeminiContent> = messages
            .into_iter()
            .map(|m| {
                let role = if m.role == "assistant" { "model" } else { "user" };
                GeminiContent {
                    role: role.to_string(),
                    parts: vec![GeminiPart::Text { text: m.content }],
                }
            })
            .collect();
            
        let request = GeminiRequest {
            contents: gemini_contents,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: Some(4096),
                response_modalities: None,
                speech_config: None,
                temperature: None,
            }),
        };
        
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}", 
            model, self.api_key
        );

        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Gemini API error {}: {}", status, error_text));
        }

        // Gemini doesn't typically send x-ratelimit headers in the same standard way as OpenAI/GitHub
        // But we check anyway if they start supporting it or if we find them.
        let mut stats = crate::types::RateLimitStats::default();
        let headers = response.headers();
        
        // Placeholder for header checking logic if simple headers existed.
        // Google APIs usually return limits in error details or specific quotas APIs, not always headers.
        
        let stream = stream! {
            let mut stream = response.bytes_stream();
            use futures::StreamExt;
            
            // Gemini stream returns a JSON array of objects, but technically in SSE mode or similar chunked transfer
            // Reqwest streams bytes. Gemini REST API streaming response is a bit different:
            // It returns a JSON array, where each chunk is a partial JSON object.
            // BUT 'streamGenerateContent' returns a stream of JSON objects if using SSE? 
            // Actually, the REST endpoint returns "[ \n { ... }, \n { ... } ]". 
            // Parsing this with a simple line-based reader is tricky.
            // However, most implementations treat it as a stream of chunks.
            // Let's implement a buffer based parser or hope it's line delimited enough.
            // Actually, Google's stream often sends one JSON object per line or chunk.
            
            // Simplified approach: accumulate buffer, try to parse JSON objects.
            
            let mut buffer = String::new();
            
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        // This is a naive parser for the array format "[{}, {}, ...]"
                        // Real Gemini output:
                        // [{
                        //   "candidates": [...]
                        // },
                        // {
                        //   ...
                        // }]
                        // We need to parse complete JSON objects.
                        
                        buffer.push_str(&text);
                        
                        // Try to find valid complete JSON objects in the buffer
                        // A simple heuristic: check for matching braces or "}," delimiters
                        
                        // For now, let's assume we can split by "\n," or similar if formatted.
                        // Better: create a proper stream parser or use lines if possible.
                        
                        // Hacky for now: Clean up the array brackets
                        if buffer.starts_with('[') {
                            buffer.remove(0);
                        }
                        
                        // Split by "}\n," or "},\n" or just "}" if it's the end
                        // This is complex to do robustly without a json stream parser.
                        // Let's try to just find full objects.
                        
                        while let Some(end_idx) = buffer.find("}\n,") 
                            .or(buffer.find("},\r\n"))
                            .or(buffer.find("}, ")) {
                                
                            let json_str = &buffer[..end_idx+1];
                            let remaining = &buffer[end_idx+1..];
                            // Clean comma
                            let remaining = remaining.trim_start_matches(|c| c == ',' || c == '\n' || c == '\r' || c == ' ');
                            
                            if let Ok(response) = serde_json::from_str::<GeminiResponse>(json_str) {
                                if let Some(candidates) = response.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        if let Some(content) = &candidate.content {
                                            if let Some(parts) = &content.parts {
                                                for part in parts {
                                                    if let Some(text) = &part.text {
                                                        yield Ok(text.clone());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            let temp_rem = remaining.to_string();
                            buffer = temp_rem;
                        }
                        
                        // Handle the last one if it ends with "}]"
                        if buffer.ends_with(']') {
                            buffer.pop(); // remove ]
                             if let Ok(response) = serde_json::from_str::<GeminiResponse>(buffer.trim()) {
                                if let Some(candidates) = response.candidates {
                                    if let Some(candidate) = candidates.first() {
                                        if let Some(content) = &candidate.content {
                                            if let Some(parts) = &content.parts {
                                                for part in parts {
                                                    if let Some(text) = &part.text {
                                                        yield Ok(text.clone());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            buffer.clear();
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

    async fn transcribe_file(
        &self,
        file_bytes: Vec<u8>,
        content_type: &str,
    ) -> Result<String, anyhow::Error> {
        // Prepare multimodal request with inline audio
        let base64_audio = BASE64.encode(file_bytes);
        
        let request = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![
                    GeminiPart::InlineData {
                        inline_data: GeminiInlineData {
                            mime_type: content_type.to_string(),
                            data: base64_audio,
                        }
                    },
                    GeminiPart::Text { 
                        text: "Transcribe this audio.".to_string() 
                    }
                ],
            }],
            generation_config: None,
        };
        
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", 
            self.default_model(), // Use default model for STT
            self.api_key
        );
        
        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await?;
            
        if !response.status().is_success() {
             let error = response.text().await?;
             return Err(anyhow::anyhow!("Gemini transcription failed: {}", error));
        }
        
        let res_json: GeminiResponse = response.json().await?;
        
        if let Some(candidates) = res_json.candidates {
            if let Some(candidate) = candidates.first() {
                if let Some(content) = &candidate.content {
                    if let Some(parts) = &content.parts {
                        for part in parts {
                            if let Some(text) = &part.text {
                                return Ok(text.clone());
                            }
                        }
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("No transcription result found"))
    }

    async fn speech(
        &self,
        text: String,
        model: Option<&str>,
        voice: Option<&str>, // e.g., "Puck", "Zephyr"
    ) -> Result<Vec<u8>, anyhow::Error> {
        let model = model.unwrap_or("gemini-2.5-flash-preview-tts"); // Default TTS model
        let voice_name = voice.unwrap_or("Puck").to_string(); // Default voice
        
        let request = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text { text }],
            }],
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: None,
                response_modalities: Some(vec!["audio".to_string()]),
                speech_config: Some(GeminiSpeechConfig {
                    multi_speaker_voice_config: None,
                    voice_config: Some(GeminiVoiceConfig {
                        prebuilt_voice_config: GeminiPrebuiltVoiceConfig {
                            voice_name,
                        }
                    }),
                }),
                temperature: None,
            }),
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}", 
            model, self.api_key
        );

        let request_json = serde_json::to_string_pretty(&request)?;
        println!("DEBUG: Sending Gemini Speech Request (Streaming):\n{}", request_json);

        let response = self.client.post(&url)
            .body(request_json)
            .header("Content-Type", "application/json")
            .send()
            .await?;

        if !response.status().is_success() {
             let error = response.text().await?;
             return Err(anyhow::anyhow!("Gemini TTS failed: {}", error));
        }
        
        // streamGenerateContent returns a JSON array of responses
        let response_text = response.text().await?;
        // println!("DEBUG: Raw Gemini TTS (Stream) Response: {}", response_text); // verbose
        
        let responses: Vec<GeminiResponse> = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("Failed to parse streaming response array: {}. Text: {}", e, response_text))?;
            
        let mut audio_buffer = Vec::new();
        let mut found_audio = false;

        for res in responses {
            if let Some(candidates) = res.candidates {
                if let Some(candidate) = candidates.first() {
                    if let Some(content) = &candidate.content {
                        if let Some(parts) = &content.parts {
                            for part in parts {
                                if let Some(inline_data) = &part.inline_data {
                                    // Decode base64 audio
                                    if let Ok(bytes) = BASE64.decode(&inline_data.data) {
                                        audio_buffer.extend_from_slice(&bytes);
                                        found_audio = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if found_audio {
            Ok(audio_buffer)
        } else {
            Err(anyhow::anyhow!("No audio content generated in stream"))
        }
    }

    async fn generate_image(
        &self,
        prompt: String,
        model: Option<&str>,
        _size: Option<&str>,
        _quality: Option<&str>,
    ) -> Result<Vec<u8>, anyhow::Error> {
        // Models: gemini-2.5-flash-image (fast), gemini-3-pro-image-preview (quality)
        let model = model.unwrap_or("gemini-2.5-flash-image");
        
        // Image generation in Gemini is text-to-image via generateContent
        // It returns inlineData with mimeType image/png (usually)
        let request = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart::Text { text: prompt }],
            }],
             generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: None,
                response_modalities: Some(vec!["image".to_string()]), // Request image output
                speech_config: None,
                temperature: None,
            }),
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", 
            model, self.api_key
        );

        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
             let error = response.text().await?;
             // Specific handling for "IMAGE_SAFETY" or similar if needed, but error text usually creates context
             return Err(anyhow::anyhow!("Gemini Image Generation failed: {}", error));
        }

        let res_json: GeminiResponse = response.json().await?;
        
        // Extract image
        if let Some(candidates) = res_json.candidates {
            if let Some(candidate) = candidates.first() {
                if let Some(content) = &candidate.content {
                    if let Some(parts) = &content.parts {
                        for part in parts {
                            if let Some(inline_data) = &part.inline_data {
                                // Potentially check mime_type here, typically "image/jpeg" or "image/png"
                                let image_bytes = BASE64.decode(&inline_data.data)?;
                                return Ok(image_bytes);
                            }
                        }
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("No image content generated (Safety filter might have triggered)"))
    }
}
