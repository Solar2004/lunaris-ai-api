# Models Endpoint

**URL**: `GET /models`

Retrieves a list of available models from all configured providers.

## Query Parameters

| Name | Type | Description |
| :--- | :--- | :--- |
| `provider` | `string` | (Optional) Filter results by provider name (e.g., `?provider=groq`). |

## Example Response

```json
{
  "providers": [
    {
      "provider": "groq",
      "default_model": "llama-3.3-70b-versatile",
      "available_models": ["llama-3.3-70b-versatile", "gemma2-9b-it"]
    }
  ]
}
```
