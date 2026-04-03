# Fine-tuning and File Management

The API now supports Mistral's fine-tuning workflow, allowing you to train custom models.

## Workflow Overview

1.  **Prepare your data**: Create a `.jsonl` file with your training examples.
2.  **Upload the file**: Use the `/files` endpoint to upload your dataset.
3.  **Create a job**: Use the `/finetuning/jobs` endpoint with the `file_id` from step 2.
4.  **Monitor**: check status via `/finetuning/jobs/:id`.

---

## 1. Upload File

`POST /files`

Uploads a file (must be `.jsonl` for fine-tuning).

### Request
- **Content-Type**: `multipart/form-data`
- **Body**: A field named `file` containing the file data.

### Example (cURL)
```bash
curl -X POST "http://localhost:3000/files" \
     -F "file=@mysamples.jsonl"
```

### Response
```json
{
  "id": "file-123456",
  "object": "file",
  "bytes": 1024,
  "created_at": 1700000000,
  "filename": "mysamples.jsonl",
  "purpose": "fine-tune"
}
```

---

## 2. Create Fine-tuning Job

`POST /finetuning/jobs`

Starts a new training job.

### Request Body
```json
{
  "model": "mistral-small-latest",
  "training_files": ["file-123456"],
  "hyperparameters": {
    "training_steps": 100
  },
  "suffix": "my-custom-model"
}
```

### Response
```json
{
  "id": "job-abc-123",
  "model": "mistral-small-latest",
  "status": "queued",
  "created_at": 1700000001
}
```

---

## 3. List Jobs

`GET /finetuning/jobs`

Returns all your fine-tuning jobs.

---

## 4. Get Job Status

`GET /finetuning/jobs/:id`

Check if your job has finished and get the `fine_tuned_model` ID.

### Response
```json
{
  "id": "job-abc-123",
  "model": "mistral-small-latest",
  "status": "succeeded",
  "created_at": 1700000001,
  "finished_at": 1700003600,
  "fine_tuned_model": "ft:mistral-small-latest:suffix:..."
}
```
