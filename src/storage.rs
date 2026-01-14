use rusqlite::{Connection, Result, params, Params};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::path::Path;

/// Simple thread-safe wrapper around SQLite connection
pub struct StorageEngine {
    conn: Arc<Mutex<Connection>>,
}

impl StorageEngine {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let conn = Connection::open(path).expect("Failed to open database");
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    pub fn execute_raw(&self, sql: &str) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        conn.execute(sql, [])
    }

    pub fn execute_raw_with_params<P: Params>(&self, sql: &str, params: P) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        conn.execute(sql, params)
    }

    pub fn query_json_with_params<P: Params>(&self, sql: &str, params: P) -> Result<Vec<Value>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(sql)?;
        
        let column_names: Vec<String> = stmt.column_names().into_iter().map(|s| s.to_string()).collect();
        
        let rows = stmt.query_map(params, |row| {
            let mut map = serde_json::Map::new();
            for (i, col_name) in column_names.iter().enumerate() {
                let val: rusqlite::types::Value = row.get(i)?;
                let json_val = match val {
                    rusqlite::types::Value::Null => Value::Null,
                    rusqlite::types::Value::Integer(i) => json!(i),
                    rusqlite::types::Value::Real(f) => json!(f),
                    rusqlite::types::Value::Text(t) => json!(t),
                    rusqlite::types::Value::Blob(_) => Value::String("<blob>".to_string()),
                };
                map.insert(col_name.clone(), json_val);
            }
            Ok(Value::Object(map))
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        
        Ok(result)
    }
}
