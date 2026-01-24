# Casper Prep Backend

FastAPI backend for analyzing Casper scenario video responses.

## Project Structure

```
casper-prep-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes
│   ├── config.py            # Settings (env vars)
│   ├── models.py            # Pydantic models for request/response
│   └── services/
│       ├── __init__.py
│       └── r2.py            # R2 download/upload utilities
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

   Or directly:
   ```bash
   python -m app.main
   ```

## API Endpoints

### Health Check
```
GET /health
```
Returns server health status.

### Trigger Analysis
```
POST /analyze/attempt/{attempt_id}
```
Triggers video analysis for a scenario attempt. Currently returns a stub response.

### Check Analysis Status
```
GET /analyze/status/{attempt_id}
```
Checks the status of an ongoing analysis. Currently returns a stub response.

## Development

The server runs on `http://localhost:8000` by default.

API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## R2 Service

The `R2Service` class provides utilities for:
- Downloading videos from R2 to temporary files
- Uploading videos to R2
- Generating video keys following the pattern: `videos/user-{userId}/scenario-attempt-{attemptId}/q{index}.webm`

Example usage:
```python
from app.services.r2 import r2_service

# Download a video
video_key = r2_service.generate_video_key("user123", 456, 0)
video_path = r2_service.download_video(video_key)

# Process video...

# Clean up
video_path.unlink()
```

## Next Steps

1. Implement actual analysis logic in `/analyze/attempt/{attempt_id}`
2. Add database integration for storing analysis results
3. Implement status tracking in `/analyze/status/{attempt_id}`
4. Add authentication/authorization
5. Set up background task processing for video analysis
