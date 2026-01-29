# 🔥 Log Anomaly Detection System

Production-ready unsupervised machine learning system for detecting anomalies in web server logs.

## 🎯 Features

- **Unsupervised Learning**: No labeled data required
- **Multiple Models**: Isolation Forest & Autoencoder support
- **Log Parsing**: Automatic parsing with Drain3 template mining
- **Feature Engineering**: 20+ engineered features
- **Real-time API**: FastAPI-based REST API
- **Explainable**: Human-readable anomaly explanations
- **Scalable**: Batch processing & streaming support
- **Production-Ready**: Logging, monitoring, error handling

## 📋 Requirements

- Python 3.8+
- 4GB RAM minimum
- Optional: CUDA-capable GPU for Autoencoder

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd log-anomaly-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your log files in `data/sample_logs.txt` or use the provided sample.

**Supported log formats:**
- Apache Combined
- Nginx
- Common Log Format
- Auto-detection

### 3. Train Model

```bash
# Train with default settings (Isolation Forest)
python training/train.py --data data/sample_logs.txt

# Train Autoencoder
python training/train.py --model autoencoder --data data/sample_logs.txt

# Limit training samples
python training/train.py --max-samples 10000
```

### 4. Run API

```bash
# Start API server
python app/main.py

# Or with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze single log
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"log": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] \"GET /api/users HTTP/1.1\" 200 1234 \"-\" \"Mozilla/5.0\""}'

# View API documentation
# Open: http://localhost:8000/docs
```

## 📊 API Endpoints

### `POST /api/v1/analyze`

Analyze a single log line.

**Request:**
```json
{
  "log": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] \"GET /admin/../../etc/passwd HTTP/1.1\" 403 0 \"-\" \"curl/7.68.0\""
}
```

**Response:**
```json
{
  "anomaly": true,
  "score": 0.87,
  "model": "isolation_forest",
  "reason": "rare log template + client error (4xx) + suspicious pattern: ../",
  "parsed_log": {
    "timestamp": "2024-01-01 12:00:00",
    "ip": "192.168.1.1",
    "method": "GET",
    "path": "/admin/../../etc/passwd",
    "status": 403,
    "template_id": 1543
  }
}
```

### `POST /api/v1/analyze/batch`

Analyze multiple logs.

**Request:**
```json
{
  "logs": [
    "log line 1",
    "log line 2"
  ]
}
```

### `GET /api/v1/health`

Health check endpoint.

### `GET /api/v1/stats`

Get detection statistics.

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
# Model selection
model:
  type: "isolation_forest"  # or "autoencoder"

# Isolation Forest parameters
isolation_forest:
  n_estimators: 200
  contamination: 0.05
  max_samples: 1000

# Autoencoder parameters
autoencoder:
  hidden_dims: [64, 32, 16, 8]
  latent_dim: 4
  epochs: 50
  batch_size: 256

# Feature engineering
features:
  suspicious_patterns:
    - "../"
    - "select"
    - "union"
    - "<script>"
    - "exec("
```

## 🔧 Advanced Usage

### Custom Log Format

```python
from app.parsers.log_parser import LogParser

parser = LogParser(config={
    "format": "custom",
    "pattern": r'(?P<ip>[\d\.]+) .* "(?P<method>\S+) (?P<path>\S+).*" (?P<status>\d+)'
})
```

### Programmatic Usage

```python
from app.core.detector import AnomalyDetector

# Initialize
detector = AnomalyDetector(model_type="isolation_forest")

# Load trained model
detector.load("models/trained/")

# Analyze
result = detector.analyze_log('192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api HTTP/1.1" 200 1234')

print(result["anomaly"])  # True/False
print(result["reason"])   # Explanation
```

### Batch Processing

```python
logs = [
    "log line 1",
    "log line 2",
    # ... thousands of logs
]

for log in logs:
    result = detector.analyze_log(log)
    if result["anomaly"]:
        print(f"ALERT: {result['reason']}")
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 📈 Performance

**Throughput:**
- Isolation Forest: ~10,000 logs/sec
- Autoencoder (CPU): ~2,000 logs/sec
- Autoencoder (GPU): ~20,000 logs/sec

**Memory:**
- Base: ~200MB
- + Model: ~50-500MB depending on model

## 🔍 How It Works

1. **Log Parsing**: Extract structured fields (IP, method, path, status, etc.)
2. **Template Mining**: Use Drain3 to identify log patterns
3. **Feature Extraction**: Generate 20+ numerical features
4. **Anomaly Detection**: Score using trained model
5. **Thresholding**: Adaptive threshold based on percentiles
6. **Explanation**: Generate human-readable reason

## 🛡️ Security Features

- Path traversal detection (`../`)
- SQL injection patterns (`union`, `select`)
- XSS detection (`<script>`)
- Command injection (`exec`, `eval`)
- Rate limiting per IP
- Brute force detection

## 📝 Logging

Logs are written to `logs/app.log` with rotation.

Configure in `.env`:
```env
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 🚨 Monitoring

Prometheus metrics available at `/metrics` (if enabled).

Key metrics:
- `anomaly_detection_requests_total`
- `anomaly_detection_anomalies_total`
- `anomaly_detection_latency_seconds`

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- Drain3 for log template mining
- Scikit-learn for Isolation Forest
- PyTorch for deep learning
- FastAPI for API framework

## 📞 Support

For issues and questions:
- GitHub Issues: <repo-url>/issues
- Documentation: <docs-url>
- Email: support@example.com

---

**Built with ❤️ for production security monitoring**
