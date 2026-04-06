# OceanMaster REST API

## Quick Start

```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Open docs
# http://localhost:8000/docs
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System status, model availability, cache age |
| POST | `/predict` | Run full prediction pipeline |
| GET | `/hotspots` | Latest hotspots (GeoJSON) |
| GET | `/hotspots/{species}` | Filter by species |
| GET | `/species` | Supported species list + parameters |
| POST | `/retrain` | Check data collection status (auto-finetune disabled) |

## Authentication

Set `OCEANMASTER_API_KEY` environment variable to enable API key auth.
Pass the key via `X-API-Key` header.

If no key is set, the API runs in open access mode (development).

## Example: Get Hotspots

```bash
curl http://localhost:8000/hotspots
```

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [131.5, 15.2]},
      "properties": {
        "rank": 1,
        "species": "yellowfin",
        "species_zh": "黃鰭鮪",
        "score": 0.85,
        "sst": 28.5,
        "cpue_index": 12.3
      }
    }
  ]
}
```

## Example: Run Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"mode": "offshore", "use_cache": true}'
```
