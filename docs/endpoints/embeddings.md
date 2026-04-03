# Embeddings Endpoint

The `/embeddings` endpoint allows you to generate text embeddings using supported providers (currently Mistral).

## Endpoint

`POST /embeddings`

## Request Body

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model` | `string` | Yes | The ID of the model to use (e.g., `mistral-embed`). |
| `input` | `array of strings` | Yes | The input text to embed. |

### Example Request

```json
{
  "model": "mistral-embed",
  "input": ["Embed this sentence.", "As well as this one."]
}
```

## Response Body

The response is a JSON object containing the embeddings and usage statistics.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `object` | `string` | Always "list". |
| `data` | `array` | A list of embedding objects. |
| `model` | `string` | The model used for generating embeddings. |
| `usage` | `object` | Usage statistics for the request. |

### Embedding Object

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `object` | `string` | Always "embedding". |
| `embedding` | `array of floats` | The embedding vector. |
| `index` | `number` | The index of the input text this embedding corresponds to. |

### Example Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.3, 0.4, ...],
      "index": 1
    }
  ],
  "model": "mistral-embed",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## Usage with cURL

```bash
curl -X POST "http://localhost:3000/embeddings" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "mistral-embed",
       "input": ["Embed this sentence."]
     }'
```
