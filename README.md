<div align="center">
  <img src="docs/banner.gif" alt="Lunaris Banner" width="100%" />
</div>

<div align="center">
  <table>
    <tr>
      <td valign="middle" align="left">
        <h1>Lunaris AI API</h1>
        <p>
          <b>The Unified AI Protocol.</b><br>
          Aggregate Groq, Mistral, OpenAI, Gemini, and more into a single, high-performance interface.<br>
          <i>Built with Rust. Designed for uptime.</i>
        </p>
        <p>
          <a href="#-features">Features</a> •
          <a href="#-api-endpoints">Endpoints</a> •
          <a href="#-usage">Usage</a>
        </p>
      </td>
      <td valign="middle" align="center">
        <img src="docs/logo.png" alt="Lunaris Logo" width="180" />
      </td>
    </tr>
  </table>

  [![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?style=flat-square&logo=rust)](https://www.rust-lang.org)
  [![Axum](https://img.shields.io/badge/Axum-0.7-blue?style=flat-square)](https://github.com/tokio-rs/axum)
  [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
</div>

---

## ✨ Features

- **🛡️ Unified Interface**: One API key, multiple providers (Groq, Mistral, Nvidia, Gemini, etc.).
- **🔄 Dynamic Model Fetching**: Always up-to-date model lists (`/models`) queried directly from upstream.
- **⚡ Advanced Fallbacks**: Zero downtime. If a provider fails, Lunaris switches instantly.
- **🧠 Code Completion (FIM)**: Specialized endpoint (`/completions`) for "Fill-In-the-Middle" IDE support.
- **🖥️ CLI Client**: Built-in terminal client for rapid testing and chat.
- **🚀 High Performance**: Built on the robust Tokio ecosystem.

## 📚 API Endpoints

### Chat Completions
`POST /chat`
```json
{
  "provider": "auto", 
  "model": "llama-3-70b",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

### Code Completions (FIM)
`POST /completions`
```json
{
  "provider": "codestral",
  "prompt": "fn add(a: i32, b: i32) {",
  "suffix": "}"
}
```

### List Models
`GET /models` or `GET /models?provider=groq`

## 🛠️ Configuration

Clone, configure `.env`, and run:

```bash
cargo run --bin lunaris-ai-api
```


