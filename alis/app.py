#!/usr/bin/env python3
"""
ALIS - Aadhaar Lifecycle Intelligence System
==============================================

Single entry point to run the complete ALIS pipeline.

Usage:
    python app.py                    # Full pipeline (load data, train, start)
    python app.py --quick            # Quick start (skip data loading)
    python app.py --train-only       # Only train models
    python app.py --dashboard-only   # Only start dashboards

This script will:
1. Initialize the database
2. Load CSV data from hackathon datasets
3. Calculate risk metrics for all pincodes
4. Train ML models (SARIMA, XGBoost, K-Means, Anomaly Detection)
5. Start the backend API server
6. Start the Streamlit dashboard
7. Open browser automatically
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
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Add backend to path
sys.path.insert(0, str(BACKEND_DIR))


def print_banner():
    """Print ALIS banner."""
    print("\n" + "=" * 70)
    print("""
     █████╗ ██╗     ██╗███████╗   v4.0
    ██╔══██╗██║     ██║██╔════╝
    ███████║██║     ██║███████╗   Aadhaar Lifecycle
    ██╔══██║██║     ██║╚════██║   Intelligence System
    ██║  ██║███████╗██║███████║
    ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   UIDAI Hackathon 2026
    """)
    print("=" * 70 + "\n")


def print_step(step_num, total, title):
    """Print step header."""
    print(f"\n{'─' * 70}")
    print(f"  Step {step_num}/{total}: {title}")
    print(f"{'─' * 70}")


def init_database():
    """Initialize database tables."""
    print("  → Creating database tables...")
    from app.database import init_database as db_init
    db_init()
    print("  ✓ Database initialized")


def check_data_exists():
    """Check if data already exists in database."""
    from app.database import SessionLocal
    from app.models.db_models import RawUpdate, PincodeMetric
    
    db = SessionLocal()
    try:
        raw_count = db.query(RawUpdate).count()
        metric_count = db.query(PincodeMetric).count()
        return raw_count > 0, raw_count, metric_count
    finally:
        db.close()


def load_csv_data():
    """Load data from CSV files."""
    import pandas as pd
    from datetime import datetime
    from sqlalchemy import text
    from app.database import engine, SessionLocal
    from app.models.db_models import RawUpdate
    
    print("  → Scanning CSV files...")
    
    bio_dir = RAW_DATA_DIR / "api_data_aadhar_biometric"
    demo_dir = RAW_DATA_DIR / "api_data_aadhar_demographic"
    enrol_dir = RAW_DATA_DIR / "api_data_aadhar_enrolment"
    
    # Load biometric data
    bio_files = list(bio_dir.glob("*.csv")) if bio_dir.exists() else []
    demo_files = list(demo_dir.glob("*.csv")) if demo_dir.exists() else []
    enrol_files = list(enrol_dir.glob("*.csv")) if enrol_dir.exists() else []
    
    print(f"  → Found: {len(bio_files)} bio, {len(demo_files)} demo, {len(enrol_files)} enrol files")
    
    if not bio_files and not demo_files and not enrol_files:
        print("  ⚠ No CSV files found in data/raw/")
        return 0
    
    # Load and merge dataframes
    def load_csvs(files, prefix):
        if not files:
            return pd.DataFrame()
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"    Warning: Could not load {f.name}: {e}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    print("  → Loading biometric data...")
    bio_df = load_csvs(bio_files, 'bio')
    
    print("  → Loading demographic data...")
    demo_df = load_csvs(demo_files, 'demo')
    
    print("  → Loading enrolment data...")
    enrol_df = load_csvs(enrol_files, 'enrol')
    
    # Prepare and rename columns
    if not bio_df.empty:
        bio_df = bio_df.rename(columns={
            'bio_age_5_17': 'bio_5_17', 'bio_age_17_': 'bio_18_plus'
        })
        bio_df['bio_0_5'] = 0
        bio_df['pincode'] = bio_df['pincode'].astype(str).str.zfill(6)
    
    if not demo_df.empty:
        demo_df = demo_df.rename(columns={
            'demo_age_5_17': 'demo_5_17', 'demo_age_17_': 'demo_18_plus'
        })
        demo_df['demo_0_5'] = 0
        demo_df['pincode'] = demo_df['pincode'].astype(str).str.zfill(6)
    
    if not enrol_df.empty:
        enrol_df = enrol_df.rename(columns={
            'age_0_5': 'enrol_0_5', 'age_5_17': 'enrol_5_17', 'age_18_greater': 'enrol_18_plus'
        })
        enrol_df['pincode'] = enrol_df['pincode'].astype(str).str.zfill(6)
    
    print("  → Merging datasets...")
    
    # Merge on date and pincode
    if not bio_df.empty and not demo_df.empty:
        merged = pd.merge(bio_df, demo_df, on=['date', 'pincode'], how='outer', suffixes=('', '_demo'))
    elif not bio_df.empty:
        merged = bio_df
    elif not demo_df.empty:
        merged = demo_df
    else:
        merged = pd.DataFrame()
    
    if not enrol_df.empty and not merged.empty:
        merged = pd.merge(merged, enrol_df, on=['date', 'pincode'], how='outer', suffixes=('', '_enrol'))
    elif not enrol_df.empty:
        merged = enrol_df
    
    if merged.empty:
        print("  ⚠ No data after merge")
        return 0
    
    # Handle state/district
    state_col = [c for c in merged.columns if 'state' in c.lower()]
    district_col = [c for c in merged.columns if 'district' in c.lower()]
    
    if state_col:
        merged['state'] = merged[state_col[0]].fillna('Unknown')
    else:
        merged['state'] = 'Unknown'
    
    if district_col:
        merged['district'] = merged[district_col[0]].fillna('')
    else:
        merged['district'] = ''
    
    # Fill NaN with 0 for numeric columns
    numeric_cols = ['bio_0_5', 'bio_5_17', 'bio_18_plus', 'demo_0_5', 'demo_5_17', 'demo_18_plus',
                   'enrol_0_5', 'enrol_5_17', 'enrol_18_plus']
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0
    
    # Calculate totals
    merged['bio_total'] = merged['bio_0_5'] + merged['bio_5_17'] + merged['bio_18_plus']
    merged['demo_total'] = merged['demo_0_5'] + merged['demo_5_17'] + merged['demo_18_plus']
    merged['enrol_total'] = merged['enrol_0_5'] + merged['enrol_5_17'] + merged['enrol_18_plus']
    merged['mobile_updates'] = 0
    
    # Clear old data
    print("  → Clearing old data...")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions"))
        conn.execute(text("DELETE FROM anomalies"))
        conn.execute(text("DELETE FROM recommendations"))
        conn.execute(text("DELETE FROM pincode_metrics"))
        conn.execute(text("DELETE FROM raw_updates"))
        conn.commit()
    
    # Insert records in batches
    print(f"  → Inserting {len(merged):,} records...")
    
    db = SessionLocal()
    batch_size = 10000
    inserted = 0
    
    for i in range(0, len(merged), batch_size):
        batch = merged.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            try:
                record = RawUpdate(
                    date=pd.to_datetime(row.get('date', datetime.now())).date(),
                    state=str(row.get('state', 'Unknown'))[:100],
                    district=str(row.get('district', ''))[:100],
                    pincode=str(row.get('pincode', '000000'))[:6],
                    bio_0_5=int(row.get('bio_0_5', 0)),
                    bio_5_17=int(row.get('bio_5_17', 0)),
                    bio_18_30=int(row.get('bio_18_plus', 0)),
                    bio_30_plus=0,
                    bio_total=int(row.get('bio_total', 0)),
                    demo_0_5=int(row.get('demo_0_5', 0)),
                    demo_5_17=int(row.get('demo_5_17', 0)),
                    demo_18_30=int(row.get('demo_18_plus', 0)),
                    demo_30_plus=0,
                    demo_total=int(row.get('demo_total', 0)),
                    mobile_updates=0,
                    enrol_0_5=int(row.get('enrol_0_5', 0)),
                    enrol_5_17=int(row.get('enrol_5_17', 0)),
                    enrol_18_plus=int(row.get('enrol_18_plus', 0)),
                    enrol_total=int(row.get('enrol_total', 0))
                )
                db.add(record)
                inserted += 1
            except Exception:
                pass
        
        db.commit()
        print(f"    Progress: {min(i+batch_size, len(merged)):,}/{len(merged):,} records")
    
    db.close()
    print(f"  ✓ Loaded {inserted:,} records")
    return inserted


def calculate_metrics():
    """Calculate risk metrics for all pincodes."""
    print("  → Calculating risk metrics...")
    
    from app.database import get_db_context
    from app.services.risk_calculator import RiskCalculatorService
    
    with get_db_context() as db:
        calculator = RiskCalculatorService(db)
        count = calculator.calculate_all_metrics(days_lookback=365)
    
    print(f"  ✓ Calculated metrics for {count:,} pincodes")
    return count


def train_models():
    """Train all ML models."""
    from app.database import get_db_context
    from app.config import settings
    from app.ml.train_models import ModelTrainer
    from app.services.clustering import ClusteringService
    
    # Train forecasting models
    print("  → Training SARIMA models...")
    trainer = ModelTrainer(settings.MODELS_DIR)
    results = trainer.train_all()
    
    if results.get('status') == 'success':
        print("  ✓ SARIMA & XGBoost models trained")
        for metric, models in results.get('models', {}).items():
            for model_name, info in models.items():
                if info.get('metadata', {}).get('in_sample_mae'):
                    print(f"    • {metric}/{model_name}: MAE={info['metadata']['in_sample_mae']:.2f}")
    else:
        print(f"  ⚠ Training issue: {results.get('status')}")
    
    # Clustering
    print("  → Running K-Means clustering...")
    with get_db_context() as db:
        clustering = ClusteringService(db)
        n_clusters = clustering.run_clustering(auto_optimize=True)
    print(f"  ✓ Created {n_clusters} clusters")
    
    # Anomaly detection (simplified)
    print("  → Running anomaly detection...")
    try:
        from app.services.anamoly_detector import AnomalyDetectorService
        with get_db_context() as db:
            detector = AnomalyDetectorService(db)
            anomaly_count = detector.run_anomaly_detection(days_lookback=90, limit_pincodes=100)
        print(f"  ✓ Detected {anomaly_count} anomalies")
    except Exception as e:
        print(f"  ⚠ Anomaly detection skipped: {e}")
    
    return results


def start_backend():
    """Start the backend API server."""
    print("  → Starting FastAPI backend on port 8000...")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    time.sleep(3)
    
    if process.poll() is None:
        print("  ✓ Backend running: http://localhost:8000")
        return process
    else:
        print("  ✗ Backend failed to start")
        return None


def start_streamlit():
    """Start the Streamlit dashboard."""
    print("  → Starting Streamlit dashboard on port 8501...")
    
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
        print("  ✓ Streamlit running: http://localhost:8501")
        return process
    else:
        print("  ✗ Streamlit failed to start")
        return None


def open_browser():
    """Open browser to dashboard."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ALIS - Aadhaar Lifecycle Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py                  # Full pipeline
    python app.py --quick          # Quick start (data already loaded)
    python app.py --train-only     # Only train models
    python app.py --dashboard-only # Only start dashboards
        """
    )
    parser.add_argument('--quick', action='store_true', help='Skip data loading (use existing data), still trains models')
    parser.add_argument('--train-only', action='store_true', help='Only train models, no server')
    parser.add_argument('--dashboard-only', action='store_true', help='Only start dashboards')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    
    args = parser.parse_args()
    
    print_banner()
    
    total_steps = 5  # Default: init, load, metrics, train, servers
    if args.dashboard_only:
        total_steps = 2  # Just start servers
    elif args.train_only:
        total_steps = 4  # init, load, metrics, train
    elif args.quick:
        total_steps = 4  # init, metrics, train, servers (skip data loading)
    
    try:
        step = 1
        
        # Step 1: Database
        if not args.dashboard_only:
            print_step(step, total_steps, "Initialize Database")
            init_database()
            step += 1
        
        # Step 2: Load Data
        if not args.quick and not args.dashboard_only and not args.train_only:
            print_step(step, total_steps, "Load CSV Data")
            has_data, raw_count, metric_count = check_data_exists()
            
            if has_data:
                print(f"  → Found existing data: {raw_count:,} records, {metric_count:,} pincodes")
                response = input("  → Reload data? This will clear existing data (y/N): ").strip().lower()
                if response == 'y':
                    load_csv_data()
                else:
                    print("  ✓ Using existing data")
            else:
                load_csv_data()
            step += 1
        
        # Step 3: Calculate Metrics
        if not args.dashboard_only:
            print_step(step, total_steps, "Calculate Risk Metrics")
            calculate_metrics()
            step += 1
        
        # Step 4: Train Models (ALWAYS run unless dashboard-only)
        if not args.dashboard_only:
            print_step(step, total_steps, "Train ML Models")
            train_models()
            step += 1
        
        # Step 5: Start Servers
        if not args.train_only:
            print_step(step, total_steps, "Start Servers")
            
            backend_process = start_backend()
            streamlit_process = start_streamlit()
            
            if backend_process and streamlit_process:
                print("\n" + "=" * 70)
                print("""
  🚀 ALIS is now running!
  
  📊 Dashboard:    http://localhost:8501
  🔌 API:          http://localhost:8000
  📚 API Docs:     http://localhost:8000/api/docs
  
  Press Ctrl+C to stop all servers
""")
                print("=" * 70)
                
                # Open browser
                if not args.no_browser:
                    threading.Thread(target=open_browser, daemon=True).start()
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n  Shutting down...")
                    backend_process.terminate()
                    streamlit_process.terminate()
                    print("  All servers stopped. Goodbye!")
            else:
                print("\n  ✗ Failed to start servers")
                return 1
        else:
            print("\n" + "=" * 70)
            print("  ✓ Training complete! Run without --train-only to start servers.")
            print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
