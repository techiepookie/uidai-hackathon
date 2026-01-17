"""
K-means clustering service for pincode segmentation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.models.db_models import PincodeMetric, PincodeCluster
from app.config import settings


# Cluster profile definitions
CLUSTER_PROFILES = {
    'HIGH_MIGRATION_URBAN': {
        'description': 'Urban areas with high migration and update activity',
        'characteristics': ['High demo updates', 'High mobile linkage', 'High volatility']
    },
    'STABLE_RURAL': {
        'description': 'Rural areas with stable, low update patterns',
        'characteristics': ['Low update rates', 'Low volatility', 'Low mobile linkage']
    },
    'CHILD_FOCUS': {
        'description': 'Areas with high child enrollment and bio update needs',
        'characteristics': ['High child bio updates', 'School areas', 'Seasonal patterns']
    },
    'HIGH_RISK': {
        'description': 'Areas requiring immediate attention',
        'characteristics': ['High bio risk', 'High demo risk', 'Low coverage']
    },
    'GROWING': {
        'description': 'Areas showing growth in enrollment and updates',
        'characteristics': ['Increasing trends', 'Moderate risk', 'Development zones']
    }
}


class ClusteringService:
    """
    Service for clustering pincodes based on update patterns.
    Uses K-means with optimal cluster selection.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
        self.min_clusters = 3
        self.max_clusters = 8
        self.default_clusters = 5
    
    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for clustering.
        
        Features:
        - Bio update rate
        - Demo update rate
        - Mobile linkage rate
        - Bio volatility
        - Demo volatility
        - Migration score
        - Child bio coverage
        """
        metrics = self.db.query(PincodeMetric).all()
        
        data = []
        for m in metrics:
            data.append({
                'pincode': m.pincode,
                'bio_update_rate': m.bio_update_rate or 0,
                'demo_update_rate': m.demo_update_rate or 0,
                'mobile_linkage_rate': m.mobile_linkage_rate or 0,
                'bio_volatility': m.bio_volatility or 0,
                'demo_volatility': m.demo_volatility or 0,
                'migration_score': m.migration_score or 0,
                'child_bio_coverage': m.child_bio_coverage or 0,
                'overall_risk_score': m.overall_risk_score or 0
            })
        
        df = pd.DataFrame(data)
        
        feature_columns = [
            'bio_update_rate', 'demo_update_rate', 'mobile_linkage_rate',
            'bio_volatility', 'demo_volatility', 'migration_score',
            'child_bio_coverage', 'overall_risk_score'
        ]
        
        return df, feature_columns
    
    def find_optimal_k(
        self,
        X: np.ndarray,
        min_k: int = None,
        max_k: int = None
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.
        """
        min_k = min_k or self.min_clusters
        max_k = max_k or self.max_clusters
        
        if len(X) < max_k:
            max_k = max(2, len(X) - 1)
        
        if len(X) < min_k:
            return min_k
        
        best_k = self.default_clusters
        best_score = -1
        
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                
                if len(set(labels)) < 2:
                    continue
                
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                logger.warning(f"K={k} failed: {e}")
        
        logger.info(f"Optimal k={best_k} with silhouette score={best_score:.3f}")
        return best_k
    
    def assign_cluster_profile(
        self,
        cluster_stats: Dict
    ) -> str:
        """
        Assign a profile name to a cluster based on its characteristics.
        """
        # Rule-based profile assignment
        if cluster_stats['avg_overall_risk'] > 70:
            return 'HIGH_RISK'
        elif cluster_stats['avg_migration_score'] > 50 and cluster_stats['avg_demo_update_rate'] > 5:
            return 'HIGH_MIGRATION_URBAN'
        elif cluster_stats['avg_child_bio_coverage'] > 50:
            return 'CHILD_FOCUS'
        elif cluster_stats['avg_bio_volatility'] < 20 and cluster_stats['avg_demo_volatility'] < 20:
            return 'STABLE_RURAL'
        else:
            return 'GROWING'
    
    def run_clustering(
        self,
        n_clusters: int = None,
        auto_optimize: bool = True
    ) -> int:
        """
        Run K-means clustering on all pincodes.
        
        Args:
            n_clusters: Number of clusters (if None, auto-optimize)
            auto_optimize: Whether to find optimal k
        
        Returns:
            Number of clusters created
        """
        logger.info("Starting clustering analysis")
        
        # Prepare data
        df, feature_columns = self.prepare_features()
        
        if len(df) < 10:
            logger.warning("Not enough data for clustering")
            return 0
        
        # Extract features and scale
        X = df[feature_columns].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of clusters
        if n_clusters is None and auto_optimize:
            n_clusters = self.find_optimal_k(X_scaled)
        else:
            n_clusters = n_clusters or self.default_clusters
        
        # Run K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        df['cluster_label'] = kmeans.fit_predict(X_scaled)
        
        # Clear existing cluster references from pincode_metrics first (to avoid FK constraint)
        self.db.query(PincodeMetric).update({'cluster_id': None})
        self.db.commit()
        
        # Now clear existing clusters
        self.db.query(PincodeCluster).delete()
        self.db.commit()
        
        # Create clusters and calculate statistics
        for cluster_id in range(n_clusters):
            cluster_data = df[df['cluster_label'] == cluster_id]
            
            cluster_stats = {
                'avg_bio_update_rate': cluster_data['bio_update_rate'].mean(),
                'avg_demo_update_rate': cluster_data['demo_update_rate'].mean(),
                'avg_mobile_linkage': cluster_data['mobile_linkage_rate'].mean(),
                'avg_bio_volatility': cluster_data['bio_volatility'].mean(),
                'avg_demo_volatility': cluster_data['demo_volatility'].mean(),
                'avg_migration_score': cluster_data['migration_score'].mean(),
                'avg_child_bio_coverage': cluster_data['child_bio_coverage'].mean(),
                'avg_overall_risk': cluster_data['overall_risk_score'].mean()
            }
            
            profile = self.assign_cluster_profile(cluster_stats)
            
            cluster = PincodeCluster(
                name=f"Cluster {cluster_id + 1}: {profile.replace('_', ' ').title()}",
                description=CLUSTER_PROFILES[profile]['description'],
                profile=profile,
                avg_bio_updates=cluster_stats['avg_bio_update_rate'],
                avg_demo_updates=cluster_stats['avg_demo_update_rate'],
                avg_mobile_updates=cluster_stats['avg_mobile_linkage'],
                pincode_count=len(cluster_data)
            )
            
            self.db.add(cluster)
            self.db.flush()
            
            # Update pincode metrics with cluster assignment
            for pincode in cluster_data['pincode'].tolist():
                self.db.query(PincodeMetric).filter(
                    PincodeMetric.pincode == pincode
                ).update({'cluster_id': cluster.id})
        
        self.db.commit()
        
        logger.info(f"Created {n_clusters} clusters")
        return n_clusters
    
    def get_cluster_summary(self) -> List[Dict]:
        """Get summary of all clusters."""
        clusters = self.db.query(PincodeCluster).all()
        
        summaries = []
        for cluster in clusters:
            # Get top pincodes in cluster
            top_pincodes = self.db.query(PincodeMetric).filter(
                PincodeMetric.cluster_id == cluster.id
            ).order_by(
                PincodeMetric.priority_score.desc()
            ).limit(5).all()
            
            summaries.append({
                'id': cluster.id,
                'name': cluster.name,
                'profile': cluster.profile,
                'description': cluster.description,
                'pincode_count': cluster.pincode_count,
                'avg_bio_updates': cluster.avg_bio_updates,
                'avg_demo_updates': cluster.avg_demo_updates,
                'avg_mobile_updates': cluster.avg_mobile_updates,
                'top_pincodes': [p.pincode for p in top_pincodes]
            })
        
        return summaries
    
    def get_similar_pincodes(
        self,
        pincode: str,
        limit: int = 10
    ) -> List[PincodeMetric]:
        """Find pincodes similar to the given one (same cluster)."""
        
        target = self.db.query(PincodeMetric).filter(
            PincodeMetric.pincode == pincode
        ).first()
        
        if not target or not target.cluster_id:
            return []
        
        similar = self.db.query(PincodeMetric).filter(
            PincodeMetric.cluster_id == target.cluster_id,
            PincodeMetric.pincode != pincode
        ).order_by(
            func.abs(PincodeMetric.overall_risk_score - target.overall_risk_score)
        ).limit(limit).all()
        
        return similar