# Completions Endpoint

**URL**: `POST /completions`

Code completion endpoint optimized for Fill-In-The-Middle (FIM) tasks.

## Body Parameters

| Name | Type | Description |
| :--- | :--- | :--- |
| `provider` | `string` | Provider name (e.g., `codestral`). |
| `prompt` | `string` | The code preceding the cursor. |
| `suffix` | `string` | (Optional) The code following the cursor. |
| `model` | `string` | (Optional) Specific model name. |

## Example Request

```json
{
  "provider": "codestral",
  "prompt": "fn calculate_sum(a: i32, b: i32) -> i32 {\n    ",
  "suffix": "\n}"
}
```
