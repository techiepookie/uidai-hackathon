# ALIS - Aadhaar Lifecycle Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.12-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

**ALIS** is a predictive intelligence system for Aadhaar lifecycle management. It transforms raw update data into actionable operational intelligence to optimize resource allocation and reduce authentication failures.

## 🚀 Features

- **Risk Scoring**: 8 risk metrics with priority scoring and categorization (LOW, MEDIUM, HIGH, CRITICAL)
- **Anomaly Detection**: Multi-method detection (Z-Score, IQR, Isolation Forest, Rolling Statistics)
- **Forecasting**: Ensemble model combining SARIMA and XGBoost for 30-90 day predictions
- **Clustering**: K-means segmentation of pincodes into operational profiles
- **Real-time Dashboard**: Interactive visualization with India map, charts, and priority tables
- **REST API**: Comprehensive FastAPI backend with 30+ endpoints

## 📋 Quick Start

### Prerequisites
- Python 3.12+
- Node.js (optional, for frontend development)
- Docker & Docker Compose (recommended)

### Option 1: Docker (Recommended)

```bash
# Clone and navigate
cd alis

# Start all services
docker-compose up -d

# Access the dashboard
open http://localhost
```

### Option 2: Local Development

```bash
# Backend setup
cd backend
pip install -r requirements.txt

# Initialize database and generate sample data
cd ../scripts
python init_db.py
python generate_sample_data.py --pincodes 100 --days 90
python load_data.py
python train_all_models.py

# Start API server
cd ../backend
uvicorn app.main:app --reload --port 8000

# Open frontend (separate terminal)
# Simply open Frontend/index.html in a browser
# Or use a local server:
cd ../Frontend
python -m http.server 3000
```

## 📁 Project Structure

```
alis/
├── backend/
│   ├── app/
│   │   ├── ml/               # Machine learning models
│   │   │   ├── ensemble.py   # Ensemble forecaster
│   │   │   ├── sarima_model.py
│   │   │   ├── xgboost_model.py
│   │   │   └── train_models.py
│   │   ├── models/           # Database & API models
│   │   │   ├── db_models.py  # SQLAlchemy ORM
│   │   │   └── schemas.py    # Pydantic schemas
│   │   ├── routers/          # API endpoints
│   │   │   ├── pincodes.py   # Pincode data
│   │   │   ├── analytics.py  # Dashboard stats
│   │   │   ├── predictions.py
│   │   │   └── anomalies.py
│   │   ├── services/         # Business logic
│   │   │   ├── risk_calculator.py
│   │   │   ├── anamoly_detector.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── forecaster.py
│   │   │   └── clustering.py
│   │   ├── config.py         # Settings
│   │   ├── database.py       # DB connection
│   │   └── main.py           # FastAPI app
│   ├── data/                 # Data storage
│   ├── requirements.txt
│   └── Dockerfile
├── Frontend/
│   ├── css/style.css         # Dark theme styles
│   ├── js/
│   │   ├── app.js            # Main application
│   │   ├── api.js            # API client
│   │   ├── charts.js         # Chart.js
│   │   ├── map.js            # Leaflet map
│   │   └── tables.js         # Data tables
│   └── index.html            # Dashboard
├── scripts/
│   ├── init_db.py            # Database setup
│   ├── generate_sample_data.py
│   ├── load_data.py
│   └── train_all_models.py
├── docker-compose.yml
├── nginx.conf
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analytics/dashboard-stats` | GET | Dashboard statistics |
| `/api/v1/analytics/state-overview` | GET | State-wise overview |
| `/api/v1/pincodes/` | GET | List all pincodes |
| `/api/v1/pincodes/priority` | GET | Top priority pincodes |
| `/api/v1/pincodes/{pincode}` | GET | Pincode details |
| `/api/v1/pincodes/{pincode}/forecast` | GET | Forecast predictions |
| `/api/v1/anomalies/` | GET | List anomalies |
| `/api/v1/predictions/{pincode}/generate` | POST | Generate forecast |
| `/health` | GET | Health check |

Full API documentation available at `http://localhost:8000/api/docs`

## ⚙️ Configuration

Create a `.env` file in the `backend/` directory:

```env
# Database
DATABASE_URL=sqlite:///./data/alis.db

# For PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost:5432/alis_db

# Application
DEBUG=true
ENVIRONMENT=development
```

## 📊 Risk Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| Child Bio Update Rate | Age 5-17 bio update frequency | High |
| Biometric Intensity | Overall bio update volume | High |
| Mobile Linkage Gap | Unlínked mobile rate | Medium |
| Demographic Update Rate | Address/demographic changes | Medium |
| Update Volatility | Standard deviation of updates | Low |
| Migration Score | Population movement indicator | Medium |
| Trend Analysis | Directional trend of updates | Medium |
| Overall Risk | Weighted composite score | - |

## 🧪 Testing

```bash
cd backend
pytest tests/ -v --cov=app
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Built for UIDAI Hackathon
</p>
