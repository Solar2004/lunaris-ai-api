# Chat Endpoint

**URL**: `POST /chat`

Conversational endpoint with streaming support.

## Body Parameters

| Name | Type | Description |
| :--- | :--- | :--- |
| `messages` | `array` | List of message objects `{ role, content }`. |
| `provider` | `string` | (Optional) Provider name (e.g., `groq`, `gemini`). Default: `auto`. |
| `model` | `string` | (Optional) Specific model name. |
| `stream` | `boolean` | (Optional) Enable streaming. |

## Example Request

```json
{
  "messages": [
    { "role": "user", "content": "Explain Rust in 2 sentences." }
  ],
  "provider": "groq",
  "stream": true
}
```

## Supported Providers
- `groq`
- `cerebras`
- `openrouter`
- `gemini`
- `codestral`
- `github`
- `nvidia`
- `mistral`
- `cohere`
