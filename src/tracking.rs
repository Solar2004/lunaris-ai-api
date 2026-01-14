use std::sync::Arc;
use crate::storage::StorageEngine;
use chrono::{Utc, DateTime};
use serde::{Serialize, Deserialize};
use crate::types::ProviderStats;

/// Tracking service for monitoring provider usage
pub struct TrackingService {
    storage: Arc<StorageEngine>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestRecord {
    pub provider: String,
    pub api_key_hash: String,  // First 8 chars for identification
    pub timestamp: DateTime<Utc>,
    pub tokens_input: u32,
    pub tokens_output: u32,
    pub latency_ms: u64,
    pub status: String,  // "success", "rate_limited", "error"
    pub model: String,
}

impl TrackingService {
    pub fn new(storage: Arc<StorageEngine>) -> Self {
        let service = Self { storage };
        service.init_tables();
        service
    }
    
    fn init_tables(&self) {
        // Table for request tracking
        let _ = self.storage.execute_raw(
            "CREATE TABLE IF NOT EXISTS request_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tokens_input INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                latency_ms INTEGER NOT NULL,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                date TEXT NOT NULL,
                minute TEXT NOT NULL
            )"
        );
        
        // Indexes for fast queries
        let _ = self.storage.execute_raw(
            "CREATE INDEX IF NOT EXISTS idx_provider_date ON request_tracking(provider, date)"
        );
        let _ = self.storage.execute_raw(
            "CREATE INDEX IF NOT EXISTS idx_provider_minute ON request_tracking(provider, minute)"
        );
        let _ = self.storage.execute_raw(
            "CREATE INDEX IF NOT EXISTS idx_api_key_date ON request_tracking(api_key_hash, date)"
        );
        
        tracing::info!("📊 Tracking tables initialized");
    }
    
    /// Record a new request
    pub fn record_request(&self, record: RequestRecord) {
        let date = record.timestamp.format("%Y-%m-%d").to_string();
        let minute = record.timestamp.format("%Y-%m-%d %H:%M").to_string();
        
        let _ = self.storage.execute_raw_with_params(
            "INSERT INTO request_tracking 
             (provider, api_key_hash, timestamp, tokens_input, tokens_output, latency_ms, status, model, date, minute)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            &[
                &record.provider as &dyn rusqlite::ToSql,
                &record.api_key_hash,
                &record.timestamp.to_rfc3339(),
                &record.tokens_input,
                &record.tokens_output,
                &record.latency_ms,
                &record.status,
                &record.model,
                &date,
                &minute,
            ]
        );
    }
    
    /// Get requests in current minute for a provider
    pub fn get_requests_this_minute(&self, provider: &str) -> u32 {
        let current_minute = Utc::now().format("%Y-%m-%d %H:%M").to_string();
        
        let result = self.storage.query_json_with_params(
            "SELECT COUNT(*) as count FROM request_tracking 
             WHERE provider = ? AND minute = ?",
            &[&provider as &dyn rusqlite::ToSql, &current_minute]
        );
        
        if let Ok(rows) = result {
            if let Some(row) = rows.first() {
                if let Some(count) = row.get("count").and_then(|v| v.as_i64()) {
                    return count as u32;
                }
            }
        }
        
        0
    }
    
    /// Get requests today for a provider
    pub fn get_requests_today(&self, provider: &str) -> u32 {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        let result = self.storage.query_json_with_params(
            "SELECT COUNT(*) as count FROM request_tracking 
             WHERE provider = ? AND date = ?",
            &[&provider as &dyn rusqlite::ToSql, &today]
        );
        
        if let Ok(rows) = result {
            if let Some(row) = rows.first() {
                if let Some(count) = row.get("count").and_then(|v| v.as_i64()) {
                    return count as u32;
                }
            }
        }
        
        0
    }
    
    /// Get requests today for a specific API key
    pub fn get_requests_today_for_key(&self, api_key_hash: &str) -> u32 {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        let result = self.storage.query_json_with_params(
            "SELECT COUNT(*) as count FROM request_tracking 
             WHERE api_key_hash = ? AND date = ?",
            &[&api_key_hash as &dyn rusqlite::ToSql, &today]
        );
        
        if let Ok(rows) = result {
            if let Some(row) = rows.first() {
                if let Some(count) = row.get("count").and_then(|v| v.as_i64()) {
                    return count as u32;
                }
            }
        }
        
        0
    }
    
    /// Get average latency for a provider today
    pub fn get_avg_latency_today(&self, provider: &str) -> u64 {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        let result = self.storage.query_json_with_params(
            "SELECT AVG(latency_ms) as avg_latency FROM request_tracking 
             WHERE provider = ? AND date = ? AND status = 'success'",
            &[&provider as &dyn rusqlite::ToSql, &today]
        );
        
        if let Ok(rows) = result {
            if let Some(row) = rows.first() {
                if let Some(avg) = row.get("avg_latency").and_then(|v| v.as_f64()) {
                    return avg as u64;
                }
            }
        }
        
        0
    }
    
    /// Get error count for a provider today
    pub fn get_error_count_today(&self, provider: &str) -> u32 {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        
        let result = self.storage.query_json_with_params(
            "SELECT COUNT(*) as count FROM request_tracking 
             WHERE provider = ? AND date = ? AND status != 'success'",
            &[&provider as &dyn rusqlite::ToSql, &today]
        );
        
        if let Ok(rows) = result {
            if let Some(row) = rows.first() {
                if let Some(count) = row.get("count").and_then(|v| v.as_i64()) {
                    return count as u32;
                }
            }
        }
        
        0
    }
    
    /// Get provider statistics
    pub fn get_provider_stats(&self, provider: &str, daily_limit: u32) -> ProviderStats {
        let requests_used = self.get_requests_today(provider);
        let percentage_used = if daily_limit > 0 {
            (requests_used as f64 / daily_limit as f64) * 100.0
        } else {
            0.0
        };
        
        ProviderStats {
            name: provider.to_string(),
            requests_used,
            requests_limit: daily_limit,
            percentage_used,
            avg_latency_ms: self.get_avg_latency_today(provider),
            error_count: self.get_error_count_today(provider),
        }
    }
    
    /// Check if provider can accept more requests (RPM check)
    pub fn can_accept_request(&self, provider: &str, rpm_limit: u32) -> bool {
        let current_rpm = self.get_requests_this_minute(provider);
        current_rpm < rpm_limit
    }
    
    /// Check if provider has daily quota available
    pub fn has_daily_quota(&self, provider: &str, daily_limit: u32) -> bool {
        let used = self.get_requests_today(provider);
        used < daily_limit
    }
    
    /// Get all provider stats
    pub fn get_all_stats(&self) -> Vec<ProviderStats> {
        // This will be populated with actual provider limits
        // For now, return empty vec
        Vec::new()
    }
    
    /// Clean old records (keep last 30 days)
    pub fn cleanup_old_records(&self) {
        let cutoff_date = (Utc::now() - chrono::Duration::days(30))
            .format("%Y-%m-%d")
            .to_string();
        
        let _ = self.storage.execute_raw_with_params(
            "DELETE FROM request_tracking WHERE date < ?",
            &[&cutoff_date as &dyn rusqlite::ToSql]
        );
        
        tracing::info!("🧹 Cleaned up tracking records older than {}", cutoff_date);
    }
}

/// Helper to hash API key for tracking (first 8 chars)
pub fn hash_api_key(api_key: &str) -> String {
    api_key.chars().take(8).collect()
}
