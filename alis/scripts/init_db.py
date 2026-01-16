"""
Initialize ALIS database and create all tables.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from loguru import logger


def init_database():
    """Initialize the database with all tables."""
    
    print("=" * 50)
    print("ALIS Database Initialization")
    print("=" * 50)
    
    # Import after path setup
    from app.config import settings
    from app.database import engine, Base, init_database as db_init
    from app.models.db_models import (
        RawUpdate, PincodeMetric, PincodeCluster,
        Prediction, Anomaly, Recommendation
    )
    
    print(f"\nDatabase URL: {settings.DATABASE_URL}")
    print(f"Data directory: {settings.DATA_DIR}")
    
    # Create data directories
    for directory in [settings.DATA_DIR, settings.RAW_DATA_DIR, 
                      settings.PROCESSED_DATA_DIR, settings.MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create all tables
    print("\nCreating database tables...")
    
    Base.metadata.create_all(bind=engine)
    
    # List created tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print("\nCreated tables:")
    for table in tables:
        columns = [c['name'] for c in inspector.get_columns(table)]
        print(f"  ✓ {table} ({len(columns)} columns)")
    
    print("\n" + "=" * 50)
    print("Database initialization complete!")
    print("=" * 50)
    
    return True


def reset_database():
    """Drop all tables and recreate them."""
    
    print("WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted.")
        return False
    
    from app.database import engine, Base
    
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    
    print("Recreating tables...")
    return init_database()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize ALIS database")
    parser.add_argument('--reset', action='store_true', help='Drop and recreate all tables')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_database()
    else:
        init_database()


if __name__ == "__main__":
    main()
