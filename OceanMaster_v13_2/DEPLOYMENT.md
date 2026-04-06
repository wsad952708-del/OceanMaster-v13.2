# OceanMaster v12 — Production Deployment Guide

> Complete guide for deploying OceanMaster on a production server.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04+ / Debian 11+ | Ubuntu 22.04 LTS |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB | 50 GB |
| CPU | 2 cores | 4 cores |
| Docker | 24+ | Latest |
| docker-compose | 2.20+ | Latest |
| Internet | Required | Stable broadband |

**Accounts needed**:
- **CMEMS** (free): https://data.marine.copernicus.eu/register
- **GFW** (optional, free): https://globalfishingwatch.org/our-apis/

---

## Deployment Options

### Option A: Docker Compose (Recommended)

#### 1. Clone and configure

```bash
git clone https://your-repo-host.com/your-org/oceanmaster.git
cd oceanmaster
cp .env.example .env
nano .env  # Add CMEMS credentials and set OCEANMASTER_API_KEY
```

> [!CAUTION]
> **Never bake credentials into Docker images.** The `.env` file is NOT copied into the production Docker image.
> Always inject it at runtime via `--env-file .env` or `docker-compose env_file:`.
> If you accidentally build with real credentials, rebuild the image immediately and rotate all keys.
> <!-- [v12-phase11-security] -->

#### 2. Build and start

```bash
docker-compose -f docker-compose.production.yml up -d --build
```

#### 3. Verify

```bash
# Health check
curl -s http://localhost:8000/health | python3 -m json.tool

# API test (replace YOUR_KEY with your OCEANMASTER_API_KEY)
curl -s -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/sea_conditions | python3 -m json.tool
```

#### 4. Access dashboard

```
http://your-server-ip:8000/dashboard/v2
```

#### 5. View logs

```bash
docker logs -f oceanmaster-prod
```

#### 6. Update

```bash
git pull
docker-compose -f docker-compose.production.yml up -d --build
```

---

### Option B: Bare Metal

#### 1. System dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip \
    nginx certbot python3-certbot-nginx \
    libhdf5-dev libnetcdf-dev
```

#### 2. Create service user

```bash
sudo useradd --system --create-home --shell /bin/bash oceanmaster
```

#### 3. Application setup

```bash
sudo -u oceanmaster bash
cd /opt
git clone https://your-repo-host.com/your-org/oceanmaster.git
cd oceanmaster
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # Add credentials
```

#### 4. Train ML models (if not included in repo)

```bash
source /opt/oceanmaster/venv/bin/activate
cd /opt/oceanmaster
python train_and_validate.py --all
```

#### 5. Systemd service

```bash
sudo cp deploy/oceanmaster.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable oceanmaster
sudo systemctl start oceanmaster
sudo systemctl status oceanmaster
```

#### 6. Nginx reverse proxy

```bash
sudo cp deploy/nginx-oceanmaster.conf /etc/nginx/sites-available/oceanmaster
# Edit server_name in the config
sudo nano /etc/nginx/sites-available/oceanmaster
sudo ln -s /etc/nginx/sites-available/oceanmaster /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

#### 7. SSL (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com
# Auto-renewal is configured automatically
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CMEMS_USERNAME` | Yes | — | Copernicus Marine username |
| `CMEMS_PASSWORD` | Yes | — | Copernicus Marine password |
| `OCEANMASTER_API_KEY` | Yes | `dev-key-change-me` | API authentication key |
| `COASTWATCH_API_KEY` | No | — | NOAA CoastWatch API key |
| `GFW_API_KEY` | No | — | Global Fishing Watch API key |
| `FLASK_PORT` | No | `8000` | Web server port |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `OCEANMASTER_RATE_LIMIT` | No | `100` | Max requests/min per IP |
| `PREDICTION_SPECIES` | No | `yellowfin` | Default species |
| `CACHE_TTL_HOURS` | No | `6` | Satellite data cache lifetime |

---

## Cron Jobs (Automated Predictions)

```bash
sudo -u oceanmaster crontab -e
```

```cron
# Run prediction every 6 hours
0 */6 * * * cd /opt/oceanmaster && /opt/oceanmaster/venv/bin/python main_v10_3.py --run-now >> /var/log/oceanmaster/predict.log 2>&1

# Weekly model retrain (Sunday 2 AM)
0 2 * * 0 cd /opt/oceanmaster && /opt/oceanmaster/venv/bin/python train_and_validate.py --all >> /var/log/oceanmaster/train.log 2>&1

# Daily cache cleanup (3 AM)
0 3 * * * cd /opt/oceanmaster && /opt/oceanmaster/venv/bin/python cron_cleanup_pipeline.py >> /var/log/oceanmaster/cleanup.log 2>&1
```

Create log directory:

```bash
sudo mkdir -p /var/log/oceanmaster
sudo chown oceanmaster:oceanmaster /var/log/oceanmaster
```

---

## Monitoring

### Health Check

```bash
# Quick check
curl -sf http://localhost:8000/health

# Detailed system health (requires API key)
curl -s -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/system-health | python3 -m json.tool
```

### Log Files

| Source | Location |
|--------|----------|
| Application (Docker) | `docker logs oceanmaster-prod` |
| Application (systemd) | `journalctl -u oceanmaster -f` |
| Predictions | `output/` directory |
| ML Training | `models/training_report_v12.md` |
| Cron jobs | `/var/log/oceanmaster/` |

### Monitoring Script

```bash
#!/bin/bash
# monitor.sh — Simple health monitor
HEALTH=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH" != "200" ]; then
    echo "$(date) OceanMaster DOWN (HTTP $HEALTH)" >> /var/log/oceanmaster/monitor.log
    # Optional: send alert via LINE/email/Slack
fi
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CMEMS timeout` | Copernicus API slow/down | Check https://data.marine.copernicus.eu/status; retry later |
| `SST: using climatology fallback` | Live data temporarily unavailable | Normal — system uses cached climatology as backup |
| `No trained ML model` | `.pkl` files missing | Run `python train_and_validate.py --all` |
| Dashboard blank map | Leaflet CDN blocked by firewall | Allow `cdn.jsdelivr.net`, `unpkg.com` |
| High memory usage (>4GB) | Large prediction grid | Reduce `GRID_RESOLUTION` in `.env` or `config.py` |
| `ConnectionError` on startup | No internet | System requires internet for satellite data |
| 403 on API endpoints | Missing or wrong API key | Set `X-API-Key` header matching `OCEANMASTER_API_KEY` |
| Port 8000 already in use | Another process | `lsof -i :8000` or change `FLASK_PORT` |

---

## Backup & Recovery

### Backup

```bash
#!/bin/bash
# backup.sh — Backup trained models, config, and catch data
BACKUP_DIR="/var/backups/oceanmaster"
DATE=$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

tar czf $BACKUP_DIR/oceanmaster-$DATE.tar.gz \
    -C /opt/oceanmaster \
    models/ config.py .env data/catch_reports.db

# Keep last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup complete: $BACKUP_DIR/oceanmaster-$DATE.tar.gz"
```

### Restore

```bash
tar xzf /var/backups/oceanmaster/oceanmaster-YYYYMMDD.tar.gz -C /opt/oceanmaster/
sudo systemctl restart oceanmaster
```

---

## Security Checklist

- [ ] `.env` file permission set to `chmod 600`
- [ ] Non-root Docker user (`Dockerfile.production` uses `appuser`)
- [ ] Non-root systemd user (`oceanmaster`)
- [ ] Nginx rate limiting enabled
- [ ] HTTPS with valid SSL certificate (Let's Encrypt)
- [ ] No credentials in code or git history
- [ ] `OCEANMASTER_API_KEY` set to a strong random string
- [ ] Firewall: only ports 80/443 open to public (SSH on separate port)
- [ ] Regular security updates: `sudo apt update && sudo apt upgrade`

---

## Performance Tuning

### For high-traffic deployments

```bash
# Increase Uvicorn workers (docker-compose.production.yml)
CMD ["python", "-m", "uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Increase Nginx buffers
proxy_buffer_size 16k;
proxy_buffers 4 32k;
```

### For memory-constrained servers

```python
# In config.py, reduce grid resolution
resolution: float = 0.5  # from 0.25 → fewer grid points
```

---

**Last updated**: Phase 10 (v12)
