"""
ALIS - Unified Application Runner

Run this script to start the complete ALIS application:
- Initializes database
- Loads data from API (if needed)
- Trains ML models
- Starts backend and frontend servers
- Opens browser to dashboard

Usage:
    python run.py              # Full startup
    python run.py --no-load    # Skip data loading
    python run.py --no-train   # Skip model training
    python run.py --quick      # Skip both loading and training
"""

import sys
import os
import time
import argparse
import subprocess
import webbrowser
import threading
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "Frontend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Add backend to path
sys.path.insert(0, str(BACKEND_DIR))


def print_banner():
    """Print ALIS banner."""
    print("\n" + "=" * 60)
    print("""
     █████╗ ██╗     ██╗███████╗
    ██╔══██╗██║     ██║██╔════╝
    ███████║██║     ██║███████╗
    ██╔══██║██║     ██║╚════██║
    ██║  ██║███████╗██║███████║
    ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝
    
    Aadhaar Lifecycle Intelligence System
    """)
    print("=" * 60 + "\n")


def init_database():
    """Initialize database tables."""
    print("[1/6] Initializing database...")
    from app.database import init_database as db_init
    db_init()
    print("      ✓ Database initialized\n")


def check_data_exists():
    """Check if data already exists in database."""
    from app.database import SessionLocal
    from app.models.db_models import RawUpdate
    
    db = SessionLocal()
    try:
        count = db.query(RawUpdate).count()
        return count > 0, count
    finally:
        db.close()


def load_api_data(max_records=None):
    """Load data from data.gov.in API."""
    print("[2/6] Loading data from data.gov.in API...")
    
    has_data, count = check_data_exists()
    if has_data:
        print(f"      → Found {count:,} existing records")
        response = input("      → Clear and reload? (y/N): ").strip().lower()
        if response != 'y':
            print("      ✓ Using existing data\n")
            return
        
        # Clear old data
        print("      → Clearing old data...")
        from sqlalchemy import text
        from app.database import engine
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM predictions"))
            conn.execute(text("DELETE FROM anomalies"))
            conn.execute(text("DELETE FROM recommendations"))
            conn.execute(text("DELETE FROM pincode_metrics"))
            conn.execute(text("DELETE FROM raw_updates"))
            conn.commit()
    
    # Load from API
    from app.database import get_db_context
    from app.services.data_ingestion import DataIngestionService
    from app.config import settings
    
    print("      → Fetching from API (this may take a while)...")
    
    with get_db_context() as db:
        ingestion = DataIngestionService(db)
        result = ingestion.ingest_from_api(
            max_records=max_records,
            batch_size=settings.API_FETCH_LIMIT
        )
        
        print(f"      ✓ Loaded {result['total_fetched']:,} records")
        print(f"      ✓ New: {result['ingested']:,}, Updated: {result['updated']:,}\n")


def calculate_metrics():
    """Calculate risk metrics for all pincodes."""
    print("[3/6] Calculating risk metrics...")
    
    from app.database import get_db_context
    from app.services.risk_calculator import RiskCalculatorService
    
    with get_db_context() as db:
        calculator = RiskCalculatorService(db)
        count = calculator.calculate_all_metrics(days_lookback=365)
        print(f"      ✓ Calculated metrics for {count} pincodes\n")


def train_models():
    """Train all ML models."""
    print("[4/6] Training ML models...")
    
    from app.database import get_db_context
    from app.config import settings
    from app.ml.train_models import ModelTrainer
    from app.services.clustering import ClusteringService
    from app.services.anamoly_detector import AnomalyDetectorService
    from app.services.forecaster import ForecasterService
    
    with get_db_context() as db:
        # Train forecasting models
        print("      → Training forecasting models...")
        trainer = ModelTrainer(settings.MODELS_DIR)
        results = trainer.train_all()
        
        if results.get('status') == 'success':
            print("      ✓ Forecasting models trained")
        else:
            print(f"      ⚠ Forecasting: {results.get('status', 'unknown')}")
        
        # Run clustering
        print("      → Running K-Means clustering...")
        clustering = ClusteringService(db)
        n_clusters = clustering.run_clustering(auto_optimize=True)
        print(f"      ✓ Created {n_clusters} clusters")
        
        # Anomaly detection
        print("      → Running anomaly detection...")
        detector = AnomalyDetectorService(db)
        anomaly_count = detector.run_anomaly_detection(days_lookback=90)
        print(f"      ✓ Detected {anomaly_count} anomalies")
        
        # Generate forecasts
        print("      → Generating forecasts...")
        forecaster = ForecasterService(db)
        forecast_count = forecaster.generate_all_forecasts(limit_pincodes=50)
        print(f"      ✓ Generated forecasts for {forecast_count} pincodes\n")


def start_backend():
    """Start the backend API server."""
    print("[5/6] Starting backend API server...")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    # Wait for server to start
    time.sleep(3)
    
    if process.poll() is None:
        print("      ✓ Backend running at http://localhost:8000")
        print("      ✓ API docs at http://localhost:8000/api/docs\n")
        return process
    else:
        print("      ✗ Backend failed to start\n")
        return None


def start_frontend():
    """Start the landing page server."""
    print("[6/7] Starting landing page server...")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "3000"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    time.sleep(2)
    
    if process.poll() is None:
        print("      ✓ Landing page running at http://localhost:3000\n")
        return process
    else:
        print("      ✗ Landing page failed to start\n")
        return None


def start_streamlit():
    """Start the Streamlit dashboard."""
    print("[7/7] Starting Streamlit dashboard...")
    
    streamlit_path = PROJECT_ROOT / "streamlit_app.py"
    
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_path), 
         "--server.port", "8501", "--server.headless", "true"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    time.sleep(4)
    
    if process.poll() is None:
        print("      ✓ Streamlit dashboard running at http://localhost:8501\n")
        return process
    else:
        print("      ✗ Streamlit failed to start\n")
        return None


def open_browser():
    """Open browser to Streamlit dashboard."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ALIS - Unified Application Runner")
    parser.add_argument('--no-load', action='store_true', help='Skip data loading')
    parser.add_argument('--no-train', action='store_true', help='Skip model training')
    parser.add_argument('--quick', action='store_true', help='Skip both loading and training')
    parser.add_argument('--max-records', type=int, default=None, help='Max records to load')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    
    args = parser.parse_args()
    
    if args.quick:
        args.no_load = True
        args.no_train = True
    
    print_banner()
    
    try:
        # Step 1: Initialize database
        init_database()
        
        # Step 2: Load data
        if not args.no_load:
            load_api_data(max_records=args.max_records)
        else:
            print("[2/6] Skipping data loading (--no-load)\n")
        
        # Step 3: Calculate metrics
        if not args.no_load:
            calculate_metrics()
        else:
            print("[3/6] Skipping metrics calculation\n")
        
        # Step 4: Train models
        if not args.no_train:
            train_models()
        else:
            print("[4/6] Skipping model training (--no-train)\n")
        
        # Step 5: Start backend
        backend_process = start_backend()
        
        # Step 6: Start frontend landing page
        frontend_process = start_frontend()
        
        # Step 7: Start Streamlit dashboard
        streamlit_process = start_streamlit()
        
        if backend_process and frontend_process and streamlit_process:
            print("=" * 60)
            print("  ALIS is now running!")
            print("")
            print("  Landing Page:      http://localhost:3000")
            print("  Streamlit Dash:    http://localhost:8501")
            print("  Backend API:       http://localhost:8000")
            print("  API Docs:          http://localhost:8000/api/docs")
            print("=" * 60)
            print("\n  Press Ctrl+C to stop all servers\n")
            
            # Open browser to Streamlit dashboard
            if not args.no_browser:
                threading.Thread(target=open_browser, daemon=True).start()
            
            # Keep running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                backend_process.terminate()
                frontend_process.terminate()
                streamlit_process.terminate()
                print("All servers stopped.")
        else:
            print("Failed to start servers. Check logs above.")
            return 1
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
