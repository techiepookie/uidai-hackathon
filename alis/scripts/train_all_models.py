"""
Train all ML models for ALIS.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def train_all_models(run_clustering: bool = True, run_anomaly_detection: bool = True):
    """
    Train all ML models and run analysis.
    """
    from app.config import settings
    from app.database import get_db_context
    from app.ml.train_models import ModelTrainer
    from app.services.clustering import ClusteringService
    from app.services.anamoly_detector import AnomalyDetectorService
    from app.services.forecaster import ForecasterService
    
    print("=" * 50)
    print("ALIS Model Training Pipeline")
    print("=" * 50)
    
    with get_db_context() as db:
        # Check if we have data
        from app.models.db_models import PincodeMetric
        pincode_count = db.query(PincodeMetric).count()
        
        if pincode_count == 0:
            print("\nError: No pincode metrics found!")
            print("Run the following first:")
            print("  1. python scripts/init_db.py")
            print("  2. python scripts/generate_sample_data.py")
            print("  3. python scripts/load_data.py")
            return False
        
        print(f"\nFound {pincode_count} pincodes with metrics")
        
        # 1. Train forecasting models
        print("\n" + "-" * 40)
        print("Training Forecasting Models")
        print("-" * 40)
        
        trainer = ModelTrainer(settings.MODELS_DIR)
        results = trainer.train_all()
        
        if results.get('status') == 'insufficient_data':
            print(f"⚠ Insufficient data for forecasting ({results.get('available_samples')} samples)")
        else:
            print(f"✓ Training complete with {results.get('samples')} samples")
            for metric, models in results.get('models', {}).items():
                for model_name, model_result in models.items():
                    status = '✓' if model_result.get('status') == 'success' else '✗'
                    print(f"  {status} {metric}/{model_name}")
        
        # 2. Run clustering
        if run_clustering:
            print("\n" + "-" * 40)
            print("Running K-Means Clustering")
            print("-" * 40)
            
            clustering = ClusteringService(db)
            n_clusters = clustering.run_clustering(auto_optimize=True)
            
            if n_clusters > 0:
                print(f"✓ Created {n_clusters} clusters")
                
                # Show cluster summary
                summary = clustering.get_cluster_summary()
                for cluster in summary:
                    print(f"  • {cluster['name']}: {cluster['pincode_count']} pincodes")
            else:
                print("⚠ Clustering failed (insufficient data)")
        
        # 3. Run anomaly detection
        if run_anomaly_detection:
            print("\n" + "-" * 40)
            print("Running Anomaly Detection")
            print("-" * 40)
            
            detector = AnomalyDetectorService(db)
            anomaly_count = detector.run_anomaly_detection(days_lookback=90)
            print(f"✓ Detected {anomaly_count} anomalies")
        
        # 4. Generate forecasts for top pincodes
        print("\n" + "-" * 40)
        print("Generating Sample Forecasts")
        print("-" * 40)
        
        forecaster = ForecasterService(db)
        # Only generate for top 10 to save time
        forecast_count = forecaster.generate_all_forecasts(limit_pincodes=10)
        print(f"✓ Generated {forecast_count} forecasts")
        
        # 5. Model evaluation
        print("\n" + "-" * 40)
        print("Model Evaluation")
        print("-" * 40)
        
        eval_results = trainer.evaluate_models(test_days=7)
        for metric, models in eval_results.items():
            print(f"\n{metric.upper()}:")
            for model_name, metrics in models.items():
                if 'error' in metrics:
                    print(f"  {model_name}: Error - {metrics['error']}")
                else:
                    print(f"  {model_name}: MAE={metrics.get('mae', 'N/A')}, RMSE={metrics.get('rmse', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("Model training pipeline complete!")
    print("=" * 50)
    print(f"\nModels saved to: {settings.MODELS_DIR}")
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all ALIS models")
    parser.add_argument('--no-clustering', action='store_true', help='Skip clustering')
    parser.add_argument('--no-anomaly', action='store_true', help='Skip anomaly detection')
    
    args = parser.parse_args()
    
    train_all_models(
        run_clustering=not args.no_clustering,
        run_anomaly_detection=not args.no_anomaly
    )


if __name__ == "__main__":
    main()
