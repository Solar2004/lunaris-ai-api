use dotenvy;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    
    // Provider API Keys (can have multiple keys separated by comma)
    pub openrouter_api_keys: Vec<String>,
    pub groq_api_keys: Vec<String>,
    pub gemini_api_keys: Vec<String>,
    pub cerebras_api_keys: Vec<String>,
    pub codestral_api_keys: Vec<String>,
    pub github_tokens: Vec<String>,
    pub nvidia_api_keys: Vec<String>,
    pub mistral_api_keys: Vec<String>,
    pub cohere_api_keys: Vec<String>,
    
    // Transcription Providers
    pub speechmatics_api_keys: Vec<String>,
    pub assemblyai_api_keys: Vec<String>,
    
    // Tracking
    pub tracking_enabled: bool,
    pub tracking_reset_hour: u8,
    
    // Cache
    pub cache_enabled: bool,
    pub cache_ttl_seconds: u64,
    
    // Alerts
    pub alert_threshold_warning: u8,
    pub alert_threshold_critical: u8,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        
        // Helper function to parse comma-separated keys
        fn parse_keys(env_var: &str) -> Vec<String> {
            std::env::var(env_var)
                .ok()
                .map(|s| s.split(',')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .collect())
                .unwrap_or_default()
        }
        
        Self {
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
            
            openrouter_api_keys: parse_keys("OPENROUTER_API_KEY"),
            groq_api_keys: parse_keys("GROQ_API_KEY"),
            gemini_api_keys: parse_keys("GEMINI_API_KEY"),
            cerebras_api_keys: parse_keys("CEREBRAS_API_KEY"),
            codestral_api_keys: parse_keys("CODESTRAL_API_KEY"),
            github_tokens: parse_keys("GITHUB_TOKEN"),
            nvidia_api_keys: parse_keys("NVIDIA_API_KEY"),
            mistral_api_keys: parse_keys("MISTRAL_API_KEY"),
            cohere_api_keys: parse_keys("COHERE_API_KEY"),
            speechmatics_api_keys: parse_keys("SPEECHMATICS_API_KEY"),
            assemblyai_api_keys: parse_keys("ASSEMBLYAI_API_KEY"),
            
            tracking_enabled: std::env::var("TRACKING_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            tracking_reset_hour: std::env::var("TRACKING_RESET_HOUR")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            
            cache_enabled: std::env::var("CACHE_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            cache_ttl_seconds: std::env::var("CACHE_TTL_SECONDS")
                .unwrap_or_else(|_| "3600".to_string())
                .parse()
                .unwrap_or(3600),
            
            alert_threshold_warning: std::env::var("ALERT_THRESHOLD_WARNING")
                .unwrap_or_else(|_| "75".to_string())
                .parse()
                .unwrap_or(75),
            alert_threshold_critical: std::env::var("ALERT_THRESHOLD_CRITICAL")
                .unwrap_or_else(|_| "90".to_string())
                .parse()
                .unwrap_or(90),
        }
    }
}
