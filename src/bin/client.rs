use std::io::{self, Write};
use serde::{Deserialize, Serialize};
use crossterm::{
    style::{Color, SetForegroundColor, ResetColor, Print, Stylize},
    execute,
    terminal::{Clear, ClearType},
};
use futures::StreamExt;
use reqwest::Client;

#[derive(Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompletionRequest {
    provider: String,
    model: Option<String>,
    prompt: String,
    suffix: Option<String>,
    stream: bool,
}

#[derive(Serialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    provider: Option<String>,
    model: Option<String>,
    stream: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let base_url = "http://127.0.0.1:3000";

    // Clear screen
    execute!(
        io::stdout(),
        Clear(ClearType::All),
        Print("🌙 Lunaris AI CLI Client\n".bold().cyan()),
        Print("----------------------\n\n")
    )?;

    // 0. Mode Selection
    print!("Select Mode (chat/completion) [chat]: ");
    io::stdout().flush()?;
    let mut mode = String::new();
    io::stdin().read_line(&mut mode)?;
    let mode = mode.trim().to_lowercase();
    let is_completion = mode == "completion" || mode == "fim";

    // 1. Select Provider
    let default_provider = if is_completion { "codestral" } else { "auto" };
    print!("Select Provider (auto/github/groq/cerebras/openrouter/gemini/mistral/codestral/nvidia/cohere) [{}]: ", default_provider);
    io::stdout().flush()?;
    let mut provider = String::new();
    io::stdin().read_line(&mut provider)?;
    let provider = provider.trim();
    let provider = if provider.is_empty() { default_provider } else { provider };

    // 2. System Prompt (Chat Only)
    let mut system_prompt = String::new();
    let mut messages = Vec::new();
    
    if !is_completion {
        print!("System Prompt [default]: ");
        io::stdout().flush()?;
        io::stdin().read_line(&mut system_prompt)?;
        let system_prompt_str = system_prompt.trim();

        if !system_prompt_str.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: system_prompt_str.to_string(),
            });
        }
    }

    // 3. Fetch and Select Model
    let model = if provider != "auto" {
        // Fetch models just for the selected provider
        let url = format!("{}/models?provider={}", base_url, provider);
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(text) = resp.text().await {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(providers) = json["providers"].as_array() {
                            if let Some(provider_info) = providers.first() {
                                let default = provider_info["default_model"].as_str().unwrap_or("");
                                let models = provider_info["available_models"]
                                    .as_array()
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str())
                                            .collect::<Vec<_>>()
                                    })
                                    .unwrap_or_default();
                                
                                if !models.is_empty() {
                                    println!("\nAvailable models for {}:", provider);
                                    for (i, model) in models.iter().enumerate() {
                                        let marker = if *model == default { " (default)" } else { "" };
                                        println!("  {}. {}{}", i + 1, model, marker);
                                    }
                                    
                                    print!("\nSelect model number or press Enter for default [{}]: ", default);
                                    io::stdout().flush()?;
                                    let mut model_choice = String::new();
                                    io::stdin().read_line(&mut model_choice)?;
                                    let model_choice = model_choice.trim();
                                    
                                    if model_choice.is_empty() {
                                        None 
                                    } else if let Ok(idx) = model_choice.parse::<usize>() {
                                        if idx > 0 && idx <= models.len() {
                                            Some(models[idx - 1].to_string())
                                        } else {
                                            println!("Invalid selection, using default");
                                            None
                                        }
                                    } else {
                                        Some(model_choice.to_string())
                                    }
                                } else {
                                    println!("⚠️  Check Server: No models returned for provider '{}'", provider);
                                    None
                                }
                            } else {
                                println!("⚠️  No models found for provider '{}'. Check if provider supported.", provider);
                                None
                            }
                        } else { None }
                    } else { None }
                } else { None }
            }
            _ => {
                println!("⚠️  Could not fetch models from API (Check if server 127.0.0.1:3000 is running), using default logic");
                None
            }
        }
    } else {
        None 
    };

    println!("\nStart typing! (Type 'exit' to quit)\n");

    loop {
        // User Input
        if is_completion {
            execute!(io::stdout(), SetForegroundColor(Color::Green), Print("Prefix (Prompt): "))?;
        } else {
            execute!(io::stdout(), SetForegroundColor(Color::Green), Print("User: "))?;
        }
        execute!(io::stdout(), ResetColor)?;
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let content = input.trim().to_string();

        if content.eq_ignore_ascii_case("exit") {
            break;
        }
        if content.is_empty() {
            continue;
        }

        let mut suffix = String::new();
        if is_completion {
             execute!(io::stdout(), SetForegroundColor(Color::Green), Print("Suffix (Optional): "))?;
             execute!(io::stdout(), ResetColor)?;
             io::stdout().flush()?;
             io::stdin().read_line(&mut suffix)?;
             suffix = suffix.trim().to_string();
        } else {
            messages.push(ChatMessage {
                role: "user".to_string(),
                content: content.clone(),
            });
        }

        // Assistant Response
        if is_completion {
            execute!(io::stdout(), SetForegroundColor(Color::Yellow), Print("Completion: "))?;
        } else {
            execute!(io::stdout(), SetForegroundColor(Color::Blue), Print("Assistant: "))?;
        }
        
        let start_time = std::time::Instant::now();
        let mut full_response = String::new();
        
        // Construct Request and URL based on mode
        let response = if is_completion {
             let request = CompletionRequest {
                provider: provider.to_string(),
                model: model.clone(),
                prompt: content.clone(),
                suffix: if suffix.is_empty() { None } else { Some(suffix) },
                stream: true,
            };
            client.post(format!("{}/completions", base_url))
                .json(&request)
                .send()
                .await
        } else {
            let request = ChatRequest {
                messages: messages.clone(),
                provider: Some(provider.to_string()),
                model: model.clone(), 
                stream: true,
            };
            client.post(format!("{}/chat", base_url))
                .json(&request)
                .send()
                .await
        };

        match response {
            Ok(res) => {
                if !res.status().is_success() {
                    let status = res.status();
                    let text = res.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                    println!("Error: Server returned {} - {}", status, text);
                    continue;
                }

                let mut stream = res.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            for line in text.lines() {
                                if line.starts_with("data: ") {
                                    let data = &line[6..];
                                    if data == "[DONE]" { continue; }
                                    print!("{}", data);
                                    io::stdout().flush()?;
                                    full_response.push_str(data);
                                }
                            }
                        }
                        Err(e) => println!("\nStream error: {}", e),
                    }
                }
            }
            Err(e) => {
                println!("Failed to connect: {}", e);
            }
        }

        execute!(io::stdout(), ResetColor)?;
        println!("\n"); 
        
        if is_completion {
            // No history for completion
        } else {
            // Add to history
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: full_response,
            });
        }

        // Stats
        let duration = start_time.elapsed();
        execute!(io::stdout(), SetForegroundColor(Color::DarkGrey))?;
        println!("(Latency: {:.2?} | Provider: {})", duration, provider);
        execute!(io::stdout(), ResetColor)?;
        println!("--------------------------------------------------");
    }

    Ok(())
}
