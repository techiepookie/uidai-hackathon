"""
Generate sample data for ALIS development and testing.
Creates realistic Aadhaar update patterns with:
- Seasonal patterns (weekly, monthly)
- Trends (increasing/decreasing)
- Random anomalies
- State/district distribution
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from faker import Faker

# Initialize Faker for Indian locale
fake = Faker('en_IN')

# Indian states and their approximate pincode ranges
STATES_CONFIG = {
    'Andhra Pradesh': {'prefix': ['50', '51', '52', '53'], 'districts': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Tirupati', 'Nellore']},
    'Bihar': {'prefix': ['80', '81', '82', '84'], 'districts': ['Patna', 'Gaya', 'Muzaffarpur', 'Bhagalpur', 'Sitamarhi']},
    'Delhi': {'prefix': ['11'], 'districts': ['Central Delhi', 'North Delhi', 'South Delhi', 'East Delhi', 'West Delhi']},
    'Gujarat': {'prefix': ['36', '37', '38', '39'], 'districts': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Gandhinagar']},
    'Karnataka': {'prefix': ['56', '57', '58', '59'], 'districts': ['Bangalore', 'Mysore', 'Hubli', 'Belgaum', 'Mangalore']},
    'Kerala': {'prefix': ['67', '68', '69'], 'districts': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 'Kollam']},
    'Madhya Pradesh': {'prefix': ['45', '46', '47', '48'], 'districts': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain']},
    'Maharashtra': {'prefix': ['40', '41', '42', '43', '44'], 'districts': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Thane']},
    'Rajasthan': {'prefix': ['30', '31', '32', '33', '34'], 'districts': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer']},
    'Tamil Nadu': {'prefix': ['60', '61', '62', '63', '64'], 'districts': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli']},
    'Uttar Pradesh': {'prefix': ['20', '21', '22', '23', '24', '25', '26', '27', '28'], 'districts': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Ghaziabad', 'Noida', 'Meerut', 'Allahabad']},
    'West Bengal': {'prefix': ['70', '71', '72', '73', '74'], 'districts': ['Kolkata', 'Howrah', 'Durgapur', 'Siliguri', 'Asansol']},
}

def generate_pincode(state: str) -> str:
    """Generate a valid-looking pincode for a state."""
    config = STATES_CONFIG[state]
    prefix = random.choice(config['prefix'])
    suffix = str(random.randint(1000, 9999))
    return prefix + suffix[:6-len(prefix)]

def generate_seasonal_pattern(n_days: int, period: int = 7) -> np.ndarray:
    """Generate a seasonal pattern (e.g., weekly)."""
    t = np.arange(n_days)
    return np.sin(2 * np.pi * t / period)

def generate_trend(n_days: int, slope: float = 0.1) -> np.ndarray:
    """Generate a linear trend."""
    return np.arange(n_days) * slope

def generate_noise(n_days: int, scale: float = 5) -> np.ndarray:
    """Generate random noise."""
    return np.random.normal(0, scale, n_days)

def generate_anomalies(n_days: int, prob: float = 0.02, magnitude: float = 3) -> np.ndarray:
    """Generate random anomaly spikes."""
    anomalies = np.zeros(n_days)
    for i in range(n_days):
        if random.random() < prob:
            anomalies[i] = magnitude * (1 if random.random() > 0.5 else -1)
    return anomalies

def generate_pincode_data(
    pincode: str,
    state: str,
    district: str,
    start_date: date,
    n_days: int = 90,
    base_population: int = None
) -> pd.DataFrame:
    """Generate synthetic update data for a single pincode."""
    
    if base_population is None:
        base_population = random.randint(5000, 50000)
    
    # Age group distribution
    age_dist = {
        '0_5': 0.08,
        '5_17': 0.18,
        '18_30': 0.25,
        '30_60': 0.35,
        '60_plus': 0.14
    }
    
    # Generate base patterns
    seasonal = generate_seasonal_pattern(n_days, period=7) * 10
    monthly_seasonal = generate_seasonal_pattern(n_days, period=30) * 5
    trend = generate_trend(n_days, slope=random.uniform(-0.1, 0.3))
    noise = generate_noise(n_days, scale=5)
    anomalies = generate_anomalies(n_days, prob=0.02, magnitude=20)
    
    # Base update rates
    bio_base = random.uniform(0.5, 2.0)  # % of population
    demo_base = random.uniform(0.3, 1.5)
    mobile_base = random.uniform(0.2, 1.0)
    
    records = []
    for i in range(n_days):
        current_date = start_date + timedelta(days=i)
        
        # Calculate daily modifiers
        modifier = 1 + 0.1 * (seasonal[i] + monthly_seasonal[i]) + 0.01 * trend[i] + 0.01 * noise[i]
        modifier = max(0.1, modifier)  # Ensure positive
        
        # Check for weekend effect
        if current_date.weekday() >= 5:  # Saturday/Sunday
            modifier *= 0.3  # Much lower on weekends
        
        # Random variation per age group
        bio_data = {}
        demo_data = {}
        enrol_data = {}
        
        for age_group, dist in age_dist.items():
            group_pop = int(base_population * dist)
            
            # Enrollments (stable)
            enrol_data[f'enrol_{age_group}'] = group_pop
            
            # Bio updates with age-specific patterns
            if age_group == '5_17':
                # Higher bio updates for children
                bio_rate = bio_base * 1.5 * modifier
            elif age_group == '60_plus':
                # Elderly have stable biometrics
                bio_rate = bio_base * 0.5 * modifier
            else:
                bio_rate = bio_base * modifier
            
            bio_data[f'bio_{age_group}'] = max(0, int(group_pop * bio_rate / 100 + noise[i] + anomalies[i]))
            
            # Demo updates
            if age_group == '18_30':
                # Young adults change address more
                demo_rate = demo_base * 2.0 * modifier
            else:
                demo_rate = demo_base * modifier
            
            demo_data[f'demo_{age_group}'] = max(0, int(group_pop * demo_rate / 100 + noise[i] * 0.5))
        
        # Mobile updates
        mobile = max(0, int(base_population * mobile_base / 100 * modifier + noise[i] * 0.3))
        
        record = {
            'date': current_date,
            'state': state,
            'district': district,
            'pincode': pincode,
            **bio_data,
            'bio_total': sum(bio_data.values()),
            **demo_data,
            'demo_total': sum(demo_data.values()),
            'mobile_updates': mobile,
            **enrol_data,
            'enrol_total': sum(enrol_data.values())
        }
        
        records.append(record)
    
    return pd.DataFrame(records)

def generate_sample_dataset(
    n_pincodes: int = 100,
    n_days: int = 90,
    output_path: Path = None
) -> pd.DataFrame:
    """Generate complete sample dataset."""
    
    print(f"Generating sample data for {n_pincodes} pincodes over {n_days} days...")
    
    # Calculate end date (today) and start date
    end_date = date.today()
    start_date = end_date - timedelta(days=n_days)
    
    all_data = []
    
    # Distribute pincodes across states
    states = list(STATES_CONFIG.keys())
    pincodes_per_state = n_pincodes // len(states)
    
    for state in states:
        config = STATES_CONFIG[state]
        districts = config['districts']
        
        for i in range(pincodes_per_state):
            pincode = generate_pincode(state)
            district = random.choice(districts)
            
            # Random base population
            base_pop = random.randint(5000, 80000)
            
            df = generate_pincode_data(
                pincode=pincode,
                state=state,
                district=district,
                start_date=start_date,
                n_days=n_days,
                base_population=base_pop
            )
            
            all_data.append(df)
        
        print(f"  Generated {pincodes_per_state} pincodes for {state}")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    if output_path is None:
        output_path = Path(__file__).parent.parent / "backend" / "data" / "raw" / "sample_data.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\nGenerated {len(combined)} records")
    print(f"Unique pincodes: {combined['pincode'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Saved to: {output_path}")
    
    return combined

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for ALIS")
    parser.add_argument('--pincodes', type=int, default=100, help='Number of pincodes')
    parser.add_argument('--days', type=int, default=90, help='Number of days of data')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    generate_sample_dataset(
        n_pincodes=args.pincodes,
        n_days=args.days,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
