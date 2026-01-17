"""
Load all hackathon CSV data (biometric, demographic, enrolment) into the database.

This script merges data from all three folders into a unified database.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def load_all_csv_data():
    """Load all CSV files from biometric, demographic, and enrolment folders."""
    from app.config import settings
    from app.database import get_db_context, init_database
    from app.models.db_models import RawUpdate
    from sqlalchemy import text
    from app.database import engine
    
    # Initialize database
    init_database()
    
    # Clear old data
    print("Clearing old data...")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM predictions"))
        conn.execute(text("DELETE FROM anomalies"))
        conn.execute(text("DELETE FROM recommendations"))
        conn.execute(text("DELETE FROM pincode_metrics"))
        conn.execute(text("DELETE FROM raw_updates"))
        conn.commit()
    print("✓ Old data cleared\n")
    
    raw_dir = settings.RAW_DATA_DIR
    
    bio_dir = raw_dir / "api_data_aadhar_biometric"
    demo_dir = raw_dir / "api_data_aadhar_demographic"
    enrol_dir = raw_dir / "api_data_aadhar_enrolment"
    
    print("=" * 60)
    print("Loading Hackathon CSV Data")
    print("=" * 60)
    
    # Load all biometric data
    print("\n[1/4] Loading Biometric data...")
    bio_dfs = []
    if bio_dir.exists():
        for csv_file in sorted(bio_dir.glob("*.csv")):
            print(f"      → {csv_file.name}")
            df = pd.read_csv(csv_file)
            bio_dfs.append(df)
        bio_df = pd.concat(bio_dfs, ignore_index=True) if bio_dfs else pd.DataFrame()
        print(f"      ✓ Loaded {len(bio_df):,} biometric records")
    else:
        bio_df = pd.DataFrame()
        print("      ⚠ Biometric folder not found")
    
    # Load all demographic data
    print("\n[2/4] Loading Demographic data...")
    demo_dfs = []
    if demo_dir.exists():
        for csv_file in sorted(demo_dir.glob("*.csv")):
            print(f"      → {csv_file.name}")
            df = pd.read_csv(csv_file)
            demo_dfs.append(df)
        demo_df = pd.concat(demo_dfs, ignore_index=True) if demo_dfs else pd.DataFrame()
        print(f"      ✓ Loaded {len(demo_df):,} demographic records")
    else:
        demo_df = pd.DataFrame()
        print("      ⚠ Demographic folder not found")
    
    # Load all enrolment data
    print("\n[3/4] Loading Enrolment data...")
    enrol_dfs = []
    if enrol_dir.exists():
        for csv_file in sorted(enrol_dir.glob("*.csv")):
            print(f"      → {csv_file.name}")
            df = pd.read_csv(csv_file)
            enrol_dfs.append(df)
        enrol_df = pd.concat(enrol_dfs, ignore_index=True) if enrol_dfs else pd.DataFrame()
        print(f"      ✓ Loaded {len(enrol_df):,} enrolment records")
    else:
        enrol_df = pd.DataFrame()
        print("      ⚠ Enrolment folder not found")
    
    # Normalize column names
    print("\n[4/4] Merging and inserting into database...")
    
    # Prepare biometric data
    if not bio_df.empty:
        bio_df.columns = bio_df.columns.str.lower().str.strip()
        # Actual columns: bio_age_5_17, bio_age_17_
        bio_df = bio_df.rename(columns={
            'bio_age_5_17': 'bio_5_17',
            'bio_age_17_': 'bio_18_plus'
        })
        bio_df['bio_0_5'] = 0  # Not in data
        bio_df['pincode'] = bio_df['pincode'].astype(str).str.zfill(6)
        bio_df['date'] = pd.to_datetime(bio_df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Prepare demographic data
    if not demo_df.empty:
        demo_df.columns = demo_df.columns.str.lower().str.strip()
        # Actual columns: demo_age_5_17, demo_age_17_
        demo_df = demo_df.rename(columns={
            'demo_age_5_17': 'demo_5_17',
            'demo_age_17_': 'demo_18_plus'
        })
        demo_df['demo_0_5'] = 0  # Not in data
        demo_df['pincode'] = demo_df['pincode'].astype(str).str.zfill(6)
        demo_df['date'] = pd.to_datetime(demo_df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Prepare enrolment data
    if not enrol_df.empty:
        enrol_df.columns = enrol_df.columns.str.lower().str.strip()
        enrol_df = enrol_df.rename(columns={
            'age_0_5': 'enrol_0_5',
            'age_5_17': 'enrol_5_17',
            'age_18_greater': 'enrol_18_plus'
        })
        enrol_df['pincode'] = enrol_df['pincode'].astype(str).str.zfill(6)
        enrol_df['date'] = pd.to_datetime(enrol_df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Create a unified dataset by grouping by date + pincode
    print("      → Grouping by date + pincode + state + district...")
    
    # Merge all three datasets
    bio_cols = ['date', 'state', 'district', 'pincode', 'bio_0_5', 'bio_5_17', 'bio_18_plus']
    demo_cols = ['date', 'pincode', 'demo_0_5', 'demo_5_17', 'demo_18_plus']
    enrol_cols = ['date', 'pincode', 'enrol_0_5', 'enrol_5_17', 'enrol_18_plus']
    
    if not bio_df.empty and not demo_df.empty:
        # First merge bio and demo
        merged = pd.merge(
            bio_df[[c for c in bio_cols if c in bio_df.columns]],
            demo_df[[c for c in demo_cols if c in demo_df.columns]],
            on=['date', 'pincode'],
            how='outer'
        )
        
        # Then merge with enrolment
        if not enrol_df.empty:
            merged = pd.merge(
                merged,
                enrol_df[[c for c in enrol_cols if c in enrol_df.columns]],
                on=['date', 'pincode'],
                how='outer'
            )
    elif not bio_df.empty:
        merged = bio_df
    elif not demo_df.empty:
        merged = demo_df
    else:
        merged = enrol_df
    
    # Fill NaN with 0 for numeric columns
    numeric_cols = ['bio_0_5', 'bio_5_17', 'bio_18_plus', 
                   'demo_0_5', 'demo_5_17', 'demo_18_plus',
                   'enrol_0_5', 'enrol_5_17', 'enrol_18_plus']
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    
    # Fill NaN for state/district (required for database)
    if 'state' in merged.columns:
        merged['state'] = merged['state'].fillna('Unknown')
    else:
        merged['state'] = 'Unknown'
    
    if 'district' in merged.columns:
        merged['district'] = merged['district'].fillna('')
    else:
        merged['district'] = ''
    
    # Calculate totals
    merged['bio_total'] = merged.get('bio_0_5', 0) + merged.get('bio_5_17', 0) + merged.get('bio_18_plus', 0)
    merged['demo_total'] = merged.get('demo_0_5', 0) + merged.get('demo_5_17', 0) + merged.get('demo_18_plus', 0)
    merged['enrol_total'] = merged.get('enrol_0_5', 0) + merged.get('enrol_5_17', 0) + merged.get('enrol_18_plus', 0)
    
    # Drop rows with invalid dates
    merged = merged.dropna(subset=['date'])
    merged['date'] = merged['date'].dt.date
    
    print(f"      → Total merged records: {len(merged):,}")
    print(f"      → Unique pincodes: {merged['pincode'].nunique():,}")
    print(f"      → Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    # Insert into database in batches
    print("      → Inserting into database...")
    
    with get_db_context() as db:
        batch_size = 10000
        total_inserted = 0
        
        for i in range(0, len(merged), batch_size):
            batch = merged.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                record = RawUpdate(
                    date=row['date'],
                    state=row.get('state', 'Unknown'),
                    district=row.get('district'),
                    pincode=row['pincode'],
                    bio_0_5=int(row.get('bio_0_5', 0)),
                    bio_5_17=int(row.get('bio_5_17', 0)),
                    bio_18_30=int(row.get('bio_18_plus', 0)) // 3,
                    bio_30_60=int(row.get('bio_18_plus', 0)) // 3,
                    bio_60_plus=int(row.get('bio_18_plus', 0)) - (int(row.get('bio_18_plus', 0)) // 3) * 2,
                    bio_total=int(row.get('bio_total', 0)),
                    demo_0_5=int(row.get('demo_0_5', 0)),
                    demo_5_17=int(row.get('demo_5_17', 0)),
                    demo_18_30=int(row.get('demo_18_plus', 0)) // 3,
                    demo_30_60=int(row.get('demo_18_plus', 0)) // 3,
                    demo_60_plus=int(row.get('demo_18_plus', 0)) - (int(row.get('demo_18_plus', 0)) // 3) * 2,
                    demo_total=int(row.get('demo_total', 0)),
                    mobile_updates=0,  # Not in CSV data
                    enrol_0_5=int(row.get('enrol_0_5', 0)),
                    enrol_5_17=int(row.get('enrol_5_17', 0)),
                    enrol_18_30=int(row.get('enrol_18_plus', 0)) // 3,
                    enrol_30_60=int(row.get('enrol_18_plus', 0)) // 3,
                    enrol_60_plus=int(row.get('enrol_18_plus', 0)) - (int(row.get('enrol_18_plus', 0)) // 3) * 2,
                    enrol_total=int(row.get('enrol_total', 0)),
                )
                db.add(record)
            
            db.commit()
            total_inserted += len(batch)
            print(f"      → Inserted {total_inserted:,} records...")
    
    print(f"\n✓ Successfully loaded {total_inserted:,} records into database!")
    
    # Now calculate metrics
    print("\n" + "-" * 40)
    print("Calculating Risk Metrics...")
    print("-" * 40)
    
    from app.services.risk_calculator import RiskCalculatorService
    with get_db_context() as db:
        calculator = RiskCalculatorService(db)
        count = calculator.calculate_all_metrics(days_lookback=365)
        print(f"✓ Calculated metrics for {count} pincodes")
    
    print("\n" + "=" * 60)
    print("CSV Data Loading Complete!")
    print("=" * 60)
    
    return total_inserted


if __name__ == "__main__":
    load_all_csv_data()
