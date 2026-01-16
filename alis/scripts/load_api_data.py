"""
Load data from data.gov.in Aadhaar API into the ALIS database.
"""

import sys
from pathlib import Path
import argparse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def load_api_data(
    state: str = None,
    district: str = None,
    max_records: int = None,
    calculate_metrics: bool = True
):
    """
    Load data from Aadhaar API.
    
    Args:
        state: Optional state filter
        district: Optional district filter
        max_records: Maximum records to fetch
        calculate_metrics: Whether to calculate metrics after loading
    """
    from app.config import settings
    from app.database import get_db_context, init_database
    from app.services.data_ingestion import DataIngestionService
    from app.services.risk_calculator import RiskCalculatorService
    
    # Initialize database tables
    init_database()
    
    print("=" * 60)
    print("ALIS - Loading from data.gov.in API")
    print("=" * 60)
    print(f"\nAPI URL: {settings.AADHAAR_API_URL}/{settings.AADHAAR_RESOURCE_ID}")
    
    if state:
        print(f"State Filter: {state}")
    if district:
        print(f"District Filter: {district}")
    if max_records:
        print(f"Max Records: {max_records}")
    
    print("\nFetching data from API...")
    
    with get_db_context() as db:
        # Ingest from API
        ingestion = DataIngestionService(db)
        
        result = ingestion.ingest_from_api(
            state_filter=state,
            district_filter=district,
            max_records=max_records,
            batch_size=settings.API_FETCH_LIMIT
        )
        
        print(f"\n✓ Total records fetched: {result['total_fetched']}")
        print(f"✓ New records: {result['ingested']}")
        print(f"✓ Updated records: {result['updated']}")
        
        # Data summary
        summary = ingestion.get_data_summary()
        print("\nData Summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Unique pincodes: {summary['unique_pincodes']}")
        print(f"  Unique states: {summary['unique_states']}")
        if summary['date_range']['start']:
            print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        # Calculate metrics if requested
        if calculate_metrics and result['ingested'] > 0:
            print("\nCalculating risk metrics...")
            calculator = RiskCalculatorService(db)
            count = calculator.calculate_all_metrics(days_lookback=90)
            print(f"✓ Calculated metrics for {count} pincodes")
    
    print("\n" + "=" * 60)
    print("API data loading complete!")
    print("=" * 60)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load Aadhaar data from data.gov.in API")
    parser.add_argument('--state', type=str, default=None, help='Filter by state name')
    parser.add_argument('--district', type=str, default=None, help='Filter by district name')
    parser.add_argument('--max-records', type=int, default=None, help='Maximum records to fetch')
    parser.add_argument('--no-metrics', action='store_true', help='Skip metric calculation')
    
    args = parser.parse_args()
    
    load_api_data(
        state=args.state,
        district=args.district,
        max_records=args.max_records,
        calculate_metrics=not args.no_metrics
    )


if __name__ == "__main__":
    main()
