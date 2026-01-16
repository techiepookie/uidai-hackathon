"""
Load data from CSV files into the ALIS database.
"""

import sys
from pathlib import Path
import argparse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def load_data(file_path: Path = None, calculate_metrics: bool = True):
    """
    Load data from CSV into database.
    
    Args:
        file_path: Path to CSV file (default: data/raw/sample_data.csv)
        calculate_metrics: Whether to calculate metrics after loading
    """
    from app.config import settings
    from app.database import get_db_context, init_database
    from app.services.data_ingestion import DataIngestionService
    from app.services.risk_calculator import RiskCalculatorService
    
    # Initialize database tables (creates them if they don't exist)
    init_database()
    
    # Default file path
    if file_path is None:
        file_path = settings.RAW_DATA_DIR / "sample_data.csv"
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        print("\nRun 'python scripts/generate_sample_data.py' first to create sample data.")
        return False
    
    print("=" * 50)
    print("ALIS Data Loading")
    print("=" * 50)
    print(f"\nLoading from: {file_path}")
    
    with get_db_context() as db:
        # Ingest data
        ingestion = DataIngestionService(db)
        
        print("\nIngesting data...")
        result = ingestion.ingest_csv(file_path)
        
        print(f"\n✓ Total rows: {result['total_rows']}")
        print(f"✓ New records: {result['ingested']}")
        print(f"✓ Updated records: {result['updated']}")
        
        # Validation report
        validation = result.get('validation', {})
        if validation.get('warnings'):
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")
        
        # Data summary
        summary = ingestion.get_data_summary()
        print("\nData Summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Unique pincodes: {summary['unique_pincodes']}")
        print(f"  Unique states: {summary['unique_states']}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        # Calculate metrics if requested
        if calculate_metrics and result['ingested'] > 0:
            print("\nCalculating risk metrics...")
            calculator = RiskCalculatorService(db)
            count = calculator.calculate_all_metrics(days_lookback=90)
            print(f"✓ Calculated metrics for {count} pincodes")
    
    print("\n" + "=" * 50)
    print("Data loading complete!")
    print("=" * 50)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load data into ALIS database")
    parser.add_argument('--file', type=str, default=None, help='Path to CSV file')
    parser.add_argument('--no-metrics', action='store_true', help='Skip metric calculation')
    
    args = parser.parse_args()
    
    file_path = Path(args.file) if args.file else None
    
    load_data(
        file_path=file_path,
        calculate_metrics=not args.no_metrics
    )


if __name__ == "__main__":
    main()
