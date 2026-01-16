"""
Data ingestion and validation service.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List, Any
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.models.db_models import RawUpdate, PincodeMetric
from app.config import settings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# State name normalization mapping
STATE_NORMALIZATION = {
    'J&K': 'Jammu and Kashmir',
    'JK': 'Jammu and Kashmir',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Chattisgarh': 'Chhattisgarh',
    'Chhatisgarh': 'Chhattisgarh',
    'Orissa': 'Odisha',
    'Pondicherry': 'Puducherry',
    'AN Islands': 'Andaman and Nicobar Islands',
    'A&N Islands': 'Andaman and Nicobar Islands',
    'D&N Haveli': 'Dadra and Nagar Haveli',
    'DNH': 'Dadra and Nagar Haveli',
    'Daman & Diu': 'Daman and Diu',
    'DD': 'Daman and Diu',
    'Delhi': 'NCT of Delhi',
    'New Delhi': 'NCT of Delhi',
    'UP': 'Uttar Pradesh',
    'MP': 'Madhya Pradesh',
    'HP': 'Himachal Pradesh',
    'TN': 'Tamil Nadu',
    'AP': 'Andhra Pradesh',
    'TS': 'Telangana',
    'WB': 'West Bengal',
    'UK': 'Uttarakhand',
    'Uttaranchal': 'Uttarakhand',
}

VALID_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
    'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Dadra and Nagar Haveli',
    'Daman and Diu', 'NCT of Delhi', 'Jammu and Kashmir', 'Ladakh',
    'Lakshadweep', 'Puducherry'
]


class DataIngestionService:
    """Service for ingesting and validating Aadhaar update data."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def normalize_state_name(self, state: str) -> str:
        """Normalize state names to standard format."""
        if not state:
            return "Unknown"
        state = state.strip()
        return STATE_NORMALIZATION.get(state, state)
    
    def validate_pincode(self, pincode: str) -> bool:
        """Validate Indian pincode format."""
        if not pincode or not isinstance(pincode, str):
            return False
        pincode = pincode.strip()
        return len(pincode) == 6 and pincode.isdigit()
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation.
        Returns validation report.
        """
        report = {
            'is_valid': True,
            'total_rows': len(df),
            'issues': [],
            'warnings': []
        }
        
        required_columns = ['date', 'state', 'pincode']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            report['is_valid'] = False
            report['issues'].append(f"Missing required columns: {missing_cols}")
            return report
        
        # Check for empty dataframe
        if len(df) == 0:
            report['is_valid'] = False
            report['issues'].append("DataFrame is empty")
            return report
        
        # Check pincode format
        invalid_pincodes = df[~df['pincode'].astype(str).str.match(r'^\d{6}$')]
        if len(invalid_pincodes) > 0:
            pct = len(invalid_pincodes) / len(df) * 100
            report['warnings'].append(f"Invalid pincodes: {len(invalid_pincodes)} ({pct:.1f}%)")
        
        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                report['warnings'].append(f"Negative values in {col}: {negative_count}")
        
        # Check for missing values
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                if pct > 5:
                    report['warnings'].append(f"High null count in {col}: {count} ({pct:.1f}%)")
        
        # Check date range
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                date_range = df['date'].max() - df['date'].min()
                report['date_range_days'] = date_range.days
            except Exception as e:
                report['warnings'].append(f"Date parsing issue: {str(e)}")
        
        report['valid_rows'] = len(df) - len(invalid_pincodes)
        report['validation_score'] = report['valid_rows'] / len(df) if len(df) > 0 else 0
        
        return report
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the dataframe."""
        df = df.copy()
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Ensure pincode is string
        if 'pincode' in df.columns:
            df['pincode'] = df['pincode'].astype(str).str.zfill(6)
        
        # Normalize state names
        if 'state' in df.columns:
            df['state'] = df['state'].apply(self.normalize_state_name)
        
        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Fill numeric columns with 0
        numeric_cols = [
            'bio_0_5', 'bio_5_17', 'bio_18_30', 'bio_30_60', 'bio_60_plus', 'bio_total',
            'demo_0_5', 'demo_5_17', 'demo_18_30', 'demo_30_60', 'demo_60_plus', 'demo_total',
            'mobile_updates',
            'enrol_0_5', 'enrol_5_17', 'enrol_18_30', 'enrol_30_60', 'enrol_60_plus', 'enrol_total'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = df[col].clip(lower=0)  # Remove negative values
        
        # Calculate totals if not present
        bio_cols = ['bio_0_5', 'bio_5_17', 'bio_18_30', 'bio_30_60', 'bio_60_plus']
        if all(col in df.columns for col in bio_cols) and 'bio_total' not in df.columns:
            df['bio_total'] = df[bio_cols].sum(axis=1)
        
        demo_cols = ['demo_0_5', 'demo_5_17', 'demo_18_30', 'demo_30_60', 'demo_60_plus']
        if all(col in df.columns for col in demo_cols) and 'demo_total' not in df.columns:
            df['demo_total'] = df[demo_cols].sum(axis=1)
        
        enrol_cols = ['enrol_0_5', 'enrol_5_17', 'enrol_18_30', 'enrol_30_60', 'enrol_60_plus']
        if all(col in df.columns for col in enrol_cols) and 'enrol_total' not in df.columns:
            df['enrol_total'] = df[enrol_cols].sum(axis=1)
        
        return df
    
    def ingest_csv(self, file_path: Path, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Ingest data from CSV file.
        Returns ingestion report.
        """
        logger.info(f"Starting ingestion from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read CSV
        df = pd.read_csv(file_path)
        logger.info(f"Read {len(df)} rows from CSV")
        
        # Validate
        validation_report = self.validate_dataframe(df)
        if not validation_report['is_valid']:
            logger.error(f"Validation failed: {validation_report['issues']}")
            raise ValueError(f"Data validation failed: {validation_report['issues']}")
        
        # Preprocess
        df = self.preprocess_dataframe(df)
        
        # Filter valid records
        df = df[df['pincode'].apply(self.validate_pincode)]
        
        # Ingest in batches
        total_ingested = 0
        total_updated = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            ingested, updated = self._ingest_batch(batch)
            total_ingested += ingested
            total_updated += updated
            logger.info(f"Processed batch {i//batch_size + 1}: {ingested} new, {updated} updated")
        
        self.db.commit()
        
        return {
            'status': 'success',
            'file': str(file_path),
            'total_rows': len(df),
            'ingested': total_ingested,
            'updated': total_updated,
            'validation': validation_report
        }
    
    def _ingest_batch(self, batch: pd.DataFrame) -> tuple:
        """Ingest a batch of records."""
        ingested = 0
        updated = 0
        
        for _, row in batch.iterrows():
            # Check if record exists
            existing = self.db.query(RawUpdate).filter(
                RawUpdate.pincode == row['pincode'],
                RawUpdate.date == row['date']
            ).first()
            
            record_data = {
                'date': row['date'],
                'state': row.get('state', 'Unknown'),
                'district': row.get('district'),
                'pincode': row['pincode'],
                'bio_0_5': row.get('bio_0_5', 0),
                'bio_5_17': row.get('bio_5_17', 0),
                'bio_18_30': row.get('bio_18_30', 0),
                'bio_30_60': row.get('bio_30_60', 0),
                'bio_60_plus': row.get('bio_60_plus', 0),
                'bio_total': row.get('bio_total', 0),
                'demo_0_5': row.get('demo_0_5', 0),
                'demo_5_17': row.get('demo_5_17', 0),
                'demo_18_30': row.get('demo_18_30', 0),
                'demo_30_60': row.get('demo_30_60', 0),
                'demo_60_plus': row.get('demo_60_plus', 0),
                'demo_total': row.get('demo_total', 0),
                'mobile_updates': row.get('mobile_updates', 0),
                'enrol_0_5': row.get('enrol_0_5', 0),
                'enrol_5_17': row.get('enrol_5_17', 0),
                'enrol_18_30': row.get('enrol_18_30', 0),
                'enrol_30_60': row.get('enrol_30_60', 0),
                'enrol_60_plus': row.get('enrol_60_plus', 0),
                'enrol_total': row.get('enrol_total', 0),
            }
            
            if existing:
                for key, value in record_data.items():
                    setattr(existing, key, value)
                updated += 1
            else:
                record = RawUpdate(**record_data)
                self.db.add(record)
                ingested += 1
        
        return ingested, updated
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of ingested data."""
        total_records = self.db.query(func.count(RawUpdate.id)).scalar()
        unique_pincodes = self.db.query(func.count(func.distinct(RawUpdate.pincode))).scalar()
        unique_states = self.db.query(func.count(func.distinct(RawUpdate.state))).scalar()
        
        date_range = self.db.query(
            func.min(RawUpdate.date),
            func.max(RawUpdate.date)
        ).first()
        
        total_bio = self.db.query(func.sum(RawUpdate.bio_total)).scalar() or 0
        total_demo = self.db.query(func.sum(RawUpdate.demo_total)).scalar() or 0
        total_mobile = self.db.query(func.sum(RawUpdate.mobile_updates)).scalar() or 0
        total_enrol = self.db.query(func.sum(RawUpdate.enrol_total)).scalar() or 0
        
        return {
            'total_records': total_records,
            'unique_pincodes': unique_pincodes,
            'unique_states': unique_states,
            'date_range': {
                'start': date_range[0].isoformat() if date_range[0] else None,
                'end': date_range[1].isoformat() if date_range[1] else None
            },
            'totals': {
                'bio_updates': total_bio,
                'demo_updates': total_demo,
                'mobile_updates': total_mobile,
                'enrollments': total_enrol
            }
        }
    
    def ingest_from_api(
        self, 
        state_filter: Optional[str] = None,
        district_filter: Optional[str] = None,
        max_records: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Ingest data from data.gov.in Aadhaar API.
        
        Args:
            state_filter: Optional state filter
            district_filter: Optional district filter
            max_records: Maximum records to fetch (None = all available)
            batch_size: Number of records per API call
            
        Returns:
            Ingestion report
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed. Run: pip install requests")
        
        logger.info("Starting API ingestion from data.gov.in")
        
        # Build base URL
        base_url = f"{settings.AADHAAR_API_URL}/{settings.AADHAAR_RESOURCE_ID}"
        
        # Use configured limit if max_records not specified
        if max_records is None:
            max_records = settings.API_MAX_RECORDS if settings.API_MAX_RECORDS > 0 else None
        
        total_ingested = 0
        total_updated = 0
        total_fetched = 0
        offset = 0
        
        while True:
            # Build request params
            params = {
                'api-key': settings.AADHAAR_API_KEY,
                'format': 'json',
                'offset': offset,
                'limit': batch_size
            }
            
            if state_filter:
                params['filters[state]'] = state_filter
            if district_filter:
                params['filters[district]'] = district_filter
            
            # Retry logic for API requests
            max_retries = 3
            retry_delay = 5
            
            for retry in range(max_retries):
                try:
                    logger.info(f"Fetching records: offset={offset}, limit={batch_size}")
                    response = requests.get(base_url, params=params, timeout=120)
                    response.raise_for_status()
                    
                    data = response.json()
                    records = data.get('records', [])
                    break  # Success, exit retry loop
                    
                except requests.Timeout:
                    if retry < max_retries - 1:
                        logger.warning(f"Request timeout, retrying in {retry_delay}s... (attempt {retry + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error("Max retries exceeded")
                        raise
                        
                except requests.RequestException as e:
                    logger.error(f"API request failed: {e}")
                    raise
            
            if not records:
                logger.info("No more records to fetch")
                break
            
            # Convert API records to DataFrame
            df = self._api_records_to_dataframe(records)
            
            if len(df) > 0:
                # Preprocess
                df = self.preprocess_dataframe(df)
                
                # Filter valid pincodes
                df = df[df['pincode'].apply(self.validate_pincode)]
                
                if len(df) > 0:
                    # Ingest batch
                    ingested, updated = self._ingest_batch(df)
                    total_ingested += ingested
                    total_updated += updated
                    self.db.commit()
            
            total_fetched += len(records)
            logger.info(f"Processed {len(records)} records (total: {total_fetched})")
            
            # Check if we've reached the limit
            if max_records and total_fetched >= max_records:
                logger.info(f"Reached max records limit: {max_records}")
                break
            
            # Check if we've fetched all available records
            total_available = int(data.get('total', 0))
            if offset + len(records) >= total_available:
                logger.info(f"Fetched all available records: {total_available}")
                break
            
            offset += batch_size
            
            # Rate limiting - add delay between requests
            import time
            time.sleep(0.5)
        
        return {
            'status': 'success',
            'source': 'data.gov.in API',
            'total_fetched': total_fetched,
            'ingested': total_ingested,
            'updated': total_updated,
            'state_filter': state_filter,
            'district_filter': district_filter
        }
    
    def _api_records_to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        """Convert API records to DataFrame with proper field mapping."""
        if not records:
            return pd.DataFrame()
        
        # Map API fields to our model
        mapped_records = []
        for rec in records:
            # Parse date (API format: DD-MM-YYYY)
            date_str = rec.get('date', '')
            try:
                date_obj = datetime.strptime(date_str, '%d-%m-%Y').date()
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid date format: {date_str}")
                    continue
            
            # Map fields
            age_0_5 = int(rec.get('age_0_5', 0))
            age_5_17 = int(rec.get('age_5_17', 0))
            age_18_plus = int(rec.get('age_18_greater', 0))
            
            mapped = {
                'date': date_obj,
                'state': rec.get('state', 'Unknown'),
                'district': rec.get('district'),
                'pincode': str(rec.get('pincode', '')).zfill(6),
                # Enrollment data from API
                'enrol_0_5': age_0_5,
                'enrol_5_17': age_5_17,
                'enrol_18_30': age_18_plus // 3,  # Distribute 18+ across age groups
                'enrol_30_60': age_18_plus // 3,
                'enrol_60_plus': age_18_plus - (age_18_plus // 3) * 2,
                'enrol_total': age_0_5 + age_5_17 + age_18_plus,
                # For enrollment data, we'll estimate bio/demo updates as a percentage
                'bio_0_5': 0,
                'bio_5_17': 0,
                'bio_18_30': 0,
                'bio_30_60': 0,
                'bio_60_plus': 0,
                'bio_total': 0,
                'demo_0_5': 0,
                'demo_5_17': 0,
                'demo_18_30': 0,
                'demo_30_60': 0,
                'demo_60_plus': 0,
                'demo_total': 0,
                'mobile_updates': 0,
            }
            mapped_records.append(mapped)
        
        return pd.DataFrame(mapped_records)