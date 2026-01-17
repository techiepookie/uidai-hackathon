# 🔐 ALIS - Aadhaar Lifecycle Intelligence System

> **AI-Powered Risk Analytics for UIDAI** | UIDAI Hackathon 2026

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Data Pipeline](#-data-pipeline)
- [ML Models](#-ml-models)
- [API Documentation](#-api-documentation)
- [Dashboard Guide](#-dashboard-guide)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contributing](#-contributing)

---

## 🎯 Overview

**ALIS** (Aadhaar Lifecycle Intelligence System) is an advanced analytics platform designed to help UIDAI monitor, predict, and optimize Aadhaar update operations across India.

### The Problem

- **Fraud Detection**: Identify unusual biometric/demographic update patterns
- **Resource Planning**: Predict enrollment center demand 30 days in advance
- **Risk Prioritization**: Focus limited resources on high-risk areas
- **Operational Efficiency**: Reduce wastage in kit deployment

### The Solution

ALIS provides:
- 🔴 **Real-time Risk Scoring** across 19,815 pincodes
- 📊 **ML-powered Forecasting** using SARIMA + XGBoost ensemble
- 🗺️ **Geographic Visualization** with interactive maps
- ⚠️ **Anomaly Detection** using Isolation Forest
- 🎯 **K-Means Clustering** for strategic segmentation

---

## ✨ Features

| Feature | Description | Technology |
|---------|-------------|------------|
| **Risk Scoring** | Multi-factor risk calculation (Bio, Demo, Mobile) | Custom Algorithm |
| **Forecasting** | 30-day bio/demo update predictions | SARIMA + XGBoost |
| **Clustering** | Auto-segmentation into 7 risk clusters | K-Means |
| **Anomaly Detection** | Spike/Drop detection using multiple methods | Isolation Forest |
| **Interactive Dashboard** | 6 pages with visualizations | Streamlit + Plotly |
| **REST API** | Full CRUD operations + ML endpoints | FastAPI |
| **Geographic Maps** | State-wise and pincode-level mapping | Plotly Mapbox |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           ALIS                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Streamlit  │  │   FastAPI    │  │   SQLite     │          │
│  │  Dashboard   │──│   Backend    │──│   Database   │          │
│  │  :8501       │  │  :8000       │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │            ML Pipeline                   │                    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │                    │
│  │  │ SARIMA  │ │ XGBoost │ │ K-Means │   │                    │
│  │  └─────────┘ └─────────┘ └─────────┘   │                    │
│  │  ┌───────────────────┐                  │                    │
│  │  │ Isolation Forest  │                  │                    │
│  │  └───────────────────┘                  │                    │
│  └─────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/alis.git
cd alis

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Place your CSV data files in:
#    backend/data/raw/api_data_aadhar_biometric/
#    backend/data/raw/api_data_aadhar_demographic/
#    backend/data/raw/api_data_aadhar_enrolment/
```

### Running ALIS

```bash
# Full pipeline (recommended for first run)
python app.py

# Quick start (skip data loading, use existing data)
python app.py --quick

# Dashboard only (after models are trained)
python app.py --dashboard-only

# Train models only (no server)
python app.py --train-only
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:8501 | Streamlit Analytics |
| **API** | http://localhost:8000 | FastAPI Backend |
| **API Docs** | http://localhost:8000/api/docs | Swagger UI |
| **Landing Page** | http://localhost:3000 | Project Overview |

---

## 📊 Data Pipeline

### Data Sources

ALIS processes three types of Aadhaar update data:

1. **Biometric Updates** (`api_data_aadhar_biometric/`)
   - Fingerprint, Iris updates by age group
   - ~1.86M records

2. **Demographic Updates** (`api_data_aadhar_demographic/`)
   - Name, Address, DOB changes
   - ~2.07M records

3. **Enrollment Data** (`api_data_aadhar_enrolment/`)
   - New Aadhaar registrations
   - ~1.0M records

### Processing Steps

```
CSV Files → Data Ingestion → Merge & Clean → Risk Calculation → ML Training
                                                      ↓
                                              Risk Metrics (19,815 pincodes)
                                              Clusters (7 groups)
                                              Anomalies (376 detected)
                                              Forecasts (SARIMA + XGBoost)
```

---

## 🧠 ML Models

### 1. Risk Scoring Engine

**Components:**
- Bio Risk Score (35% weight)
- Demo Risk Score (25% weight)
- Mobile Linkage Gap (20% weight)
- Migration Score (10% weight)
- Volatility Score (10% weight)

**Categories:**
- 🔴 **CRITICAL** (80-100): Immediate investigation
- 🟠 **HIGH** (60-79): Close monitoring
- 🟡 **MEDIUM** (40-59): Periodic review
- 🟢 **LOW** (0-39): Normal operation

### 2. Forecasting (SARIMA + XGBoost)

**SARIMA** (Seasonal ARIMA):
- Captures trends and seasonality
- Auto-optimizes (p,d,q) parameters
- AIC-based model selection

**XGBoost**:
- Gradient boosted trees
- Lag features (7, 14, 30 days)
- Rolling statistics

**Ensemble**: Weighted average of both models

### 3. K-Means Clustering

**Cluster Profiles:**
- `HIGH_RISK`: Immediate attention areas
- `HIGH_MIGRATION_URBAN`: Urban areas with high mobility
- `CHILD_FOCUS`: Schools/educational zones
- `STABLE_RURAL`: Low-activity rural areas
- `GROWING`: Developing regions

### 4. Anomaly Detection

**Methods:**
- Z-Score (statistical)
- IQR (robust)
- Isolation Forest (ML)

**Consensus**: Flags anomaly when 2+ methods agree

---

## 🔌 API Documentation

### Endpoints Overview

```
GET  /api/v1/pincodes/          # List all pincodes
GET  /api/v1/pincodes/{pincode} # Get pincode details
POST /api/v1/pincodes/calculate # Recalculate metrics

GET  /api/v1/predictions/       # List predictions
POST /api/v1/predictions/       # Generate forecast

GET  /api/v1/anomalies/         # List anomalies
GET  /api/v1/clusters/          # List clusters
GET  /api/v1/recommendations/   # Get recommendations
```

### Example: Get Pincode Risk

```bash
curl http://localhost:8000/api/v1/pincodes/110001
```

Response:
```json
{
  "pincode": "110001",
  "state": "Delhi",
  "bio_risk_score": 75.2,
  "overall_risk_score": 82.5,
  "risk_category": "CRITICAL",
  "cluster_id": 3
}
```

---

## 📈 Dashboard Guide

### Navigation

| Page | Purpose |
|------|---------|
| **🏠 Home & Tutorial** | Step-by-step guide for new users |
| **📊 Dashboard** | KPIs, Risk Distribution, Trends |
| **🗺️ Map View** | Geographic visualization |
| **📈 Analytics** | Correlation, Clusters, Distributions |
| **🧠 Model Evaluation** | MAE, RMSE, AIC metrics |
| **⚠️ Anomalies** | Detected spikes and drops |
| **🔮 Predictions** | Generate forecasts |

### Tutorial Mode

Enable/disable tutorial hints via the sidebar checkbox:
`☑️ Show Tutorial`

---

## 📁 Project Structure

```
alis/
├── app.py                 # 🚀 Main entry point
├── streamlit_app.py       # 📊 Streamlit dashboard
├── run.py                 # Alternative launcher
├── .gitignore
├── README.md
│
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI application
│   │   ├── config.py      # Configuration
│   │   ├── database.py    # SQLAlchemy setup
│   │   │
│   │   ├── models/
│   │   │   └── db_models.py   # Database models
│   │   │
│   │   ├── routers/
│   │   │   ├── pincodes.py
│   │   │   ├── predictions.py
│   │   │   ├── anomalies.py
│   │   │   └── clusters.py
│   │   │
│   │   ├── services/
│   │   │   ├── risk_calculator.py
│   │   │   ├── forecaster.py
│   │   │   ├── clustering.py
│   │   │   ├── anamoly_detector.py
│   │   │   └── data_ingestion.py
│   │   │
│   │   └── ml/
│   │       ├── sarima_model.py
│   │       ├── xgboost_model.py
│   │       ├── ensemble.py
│   │       └── train_models.py
│   │
│   ├── data/
│   │   ├── raw/           # CSV source files (gitignored)
│   │   └── models/        # Trained model files
│   │
│   ├── requirements.txt
│   └── .env               # Environment variables
│
├── Frontend/
│   └── index.html         # Landing page
│
└── scripts/
    ├── load_csv_data.py   # CSV data loader
    ├── train_all_models.py
    └── clear_data.py
```

---

## ⚙️ Configuration

### Environment Variables

Create `backend/.env`:

```env
# Database
DATABASE_URL=sqlite:///./data/alis.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# ML Settings
MODEL_DIR=./data/models
TRAINING_SAMPLES=90
```

### Key Settings (`backend/app/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `RISK_CRITICAL_THRESHOLD` | 80 | Score for critical risk |
| `RISK_HIGH_THRESHOLD` | 60 | Score for high risk |
| `DEFAULT_CLUSTERS` | 5 | K-Means clusters |
| `FORECAST_HORIZON` | 30 | Days to forecast |

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Pincodes Analyzed | 19,815 |
| Records Processed | 3.7M+ |
| Model Accuracy | 87% |
| Risk Clusters | 7 |
| Anomalies Detected | 376 |
| API Response Time | <200ms |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Create Pull Request

---

## 📜 License

This project was developed for **UIDAI Hackathon 2026**.

MIT License - See [LICENSE](LICENSE) for details.

---

## 👥 Team

**ALIS Team** - UIDAI Hackathon 2026

---

<p align="center">
  <strong>🔐 ALIS - Securing India's Digital Identity</strong><br>
  Built with ❤️ for Digital India
</p>
