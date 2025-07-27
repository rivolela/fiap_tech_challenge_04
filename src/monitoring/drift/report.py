"""
Drift Report Generator
=====================

Generates comprehensive reports from drift detection results.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class ReportGenerator:
    """Generates comprehensive drift reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self, 
        drift_results: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report
        
        Args:
            drift_results: Results from drift detection
            metadata: Additional metadata for the report
            
        Returns:
            Dict containing the formatted report
        """
        try:
            report_id = f"drift_report_{int(time.time())}"
            
            report = {
                'report_id': report_id,
                'timestamp': datetime.now().isoformat(),
                'summary': self._generate_summary(drift_results),
                'detailed_analysis': drift_results,
                'recommendations': self._generate_recommendations(drift_results),
                'metadata': metadata or {},
                'report_version': '1.0',
                'status': 'success'
            }
            
            # Add risk assessment
            report['risk_assessment'] = self._assess_risk(drift_results)
            
            # Add data quality metrics
            report['data_quality'] = self._assess_data_quality(drift_results)
            
            self.logger.info(f"Drift report generated successfully: {report_id}")
            return report
            
        except Exception as e:
            error_msg = f"Error generating drift report: {str(e)}"
            self.logger.error(error_msg)
            return {
                'report_id': f"error_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'status': 'error'
            }
    
    def _generate_summary(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of drift analysis"""
        if not drift_results or 'error' in drift_results:
            return {
                'total_features': 0,
                'features_with_drift': 0,
                'drift_percentage': 0,
                'overall_status': 'error',
                'error': drift_results.get('error', 'Unknown error')
            }
        
        total_features = len(drift_results)
        features_with_drift = 0
        high_drift_features = 0
        
        drift_scores = []
        
        for feature, results in drift_results.items():
            if isinstance(results, dict) and 'error' not in results:
                # Check if drift was detected
                if results.get('drift_detected', False):
                    features_with_drift += 1
                    
                    # Calculate drift severity
                    severity = self._calculate_drift_severity(results)
                    drift_scores.append(severity)
                    
                    if severity > 0.7:  # High drift threshold
                        high_drift_features += 1
        
        drift_percentage = (features_with_drift / total_features * 100) if total_features > 0 else 0
        
        # Determine overall status
        if drift_percentage > 60:
            status = 'critical'
        elif drift_percentage > 30:
            status = 'warning'
        elif drift_percentage > 10:
            status = 'minor_drift'
        else:
            status = 'stable'
        
        # Calculate average drift score
        avg_drift_score = sum(drift_scores) / len(drift_scores) if drift_scores else 0
        
        return {
            'total_features': total_features,
            'features_with_drift': features_with_drift,
            'high_drift_features': high_drift_features,
            'drift_percentage': round(drift_percentage, 2),
            'average_drift_score': round(avg_drift_score, 3),
            'overall_status': status,
            'status_description': self._get_status_description(status)
        }
    
    def _generate_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        if not drift_results or 'error' in drift_results:
            recommendations.append("❌ Unable to analyze drift due to data issues. Check data quality and format.")
            recommendations.append("🔍 Verify that both reference and current datasets are properly formatted.")
            return recommendations
        
        # Analyze drift patterns
        drift_features = []
        high_drift_features = []
        statistical_drift_features = []
        distribution_drift_features = []
        
        for feature, results in drift_results.items():
            if isinstance(results, dict) and 'error' not in results:
                if results.get('drift_detected', False):
                    drift_features.append(feature)
                    
                    # Check severity
                    severity = self._calculate_drift_severity(results)
                    if severity > 0.7:
                        high_drift_features.append(feature)
                    
                    # Check drift types
                    if 'statistical' in results and results['statistical'].get('drift_detected', False):
                        statistical_drift_features.append(feature)
                    
                    if 'distribution' in results and results['distribution'].get('drift_detected', False):
                        distribution_drift_features.append(feature)
        
        # Generate specific recommendations
        if not drift_features:
            recommendations.append("✅ No significant drift detected across all features.")
            recommendations.append("📊 Continue regular monitoring to maintain data quality.")
        else:
            # High-level recommendations
            recommendations.append(f"⚠️ Drift detected in {len(drift_features)} feature(s): {', '.join(drift_features[:5])}")
            
            if len(drift_features) > 5:
                recommendations.append(f"   ... and {len(drift_features) - 5} more features.")
            
            # Severity-based recommendations
            if high_drift_features:
                recommendations.append(f"🚨 HIGH PRIORITY: Features with severe drift: {', '.join(high_drift_features)}")
                recommendations.append("🔄 Immediate model retraining recommended.")
            
            # Type-specific recommendations
            if statistical_drift_features:
                recommendations.append(f"📈 Statistical drift detected in: {', '.join(statistical_drift_features[:3])}")
                recommendations.append("🔍 Investigate changes in data distribution or preprocessing pipeline.")
            
            if distribution_drift_features:
                recommendations.append(f"📊 Distribution drift detected in: {', '.join(distribution_drift_features[:3])}")
                recommendations.append("🎯 Consider updating feature engineering or data collection methods.")
            
            # General recommendations
            if len(drift_features) > len(drift_results) * 0.5:
                recommendations.append("🚨 URGENT: More than 50% of features show drift.")
                recommendations.append("🛠️ Comprehensive system review required.")
            
            recommendations.append("📅 Schedule regular drift monitoring and set up automated alerts.")
            recommendations.append("🔄 Plan for model retraining with recent data.")
        
        return recommendations
    
    def _assess_risk(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk level based on drift results"""
        if not drift_results or 'error' in drift_results:
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'risk_factors': ['Unable to assess due to data issues']
            }
        
        risk_factors = []
        risk_score = 0
        
        total_features = len(drift_results)
        drift_count = 0
        high_drift_count = 0
        
        for feature, results in drift_results.items():
            if isinstance(results, dict) and 'error' not in results:
                if results.get('drift_detected', False):
                    drift_count += 1
                    severity = self._calculate_drift_severity(results)
                    risk_score += severity
                    
                    if severity > 0.7:
                        high_drift_count += 1
        
        # Normalize risk score
        risk_score = risk_score / total_features if total_features > 0 else 0
        
        # Identify risk factors
        drift_percentage = (drift_count / total_features * 100) if total_features > 0 else 0
        
        if drift_percentage > 50:
            risk_factors.append(f"High percentage of features with drift ({drift_percentage:.1f}%)")
        
        if high_drift_count > 0:
            risk_factors.append(f"{high_drift_count} features with severe drift")
        
        if risk_score > 0.6:
            risk_factors.append("High average drift severity")
        
        # Determine risk level
        if risk_score > 0.7 or drift_percentage > 60:
            risk_level = 'critical'
        elif risk_score > 0.4 or drift_percentage > 30:
            risk_level = 'high'
        elif risk_score > 0.2 or drift_percentage > 10:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 3),
            'risk_factors': risk_factors or ['No significant risk factors identified']
        }
    
    def _assess_data_quality(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality metrics"""
        if not drift_results or 'error' in drift_results:
            return {
                'quality_score': 0,
                'issues': ['Unable to assess data quality'],
                'recommendations': ['Fix data processing issues']
            }
        
        issues = []
        total_features = len(drift_results)
        features_with_errors = 0
        
        for feature, results in drift_results.items():
            if isinstance(results, dict) and 'error' in results:
                features_with_errors += 1
                issues.append(f"Analysis failed for feature '{feature}': {results['error']}")
        
        # Calculate quality score
        quality_score = max(0, 1 - (features_with_errors / total_features)) if total_features > 0 else 0
        
        recommendations = []
        if features_with_errors > 0:
            recommendations.append("Fix data processing errors for affected features")
        
        if quality_score < 0.8:
            recommendations.append("Improve data validation and cleaning processes")
        
        if not recommendations:
            recommendations.append("Data quality appears good")
        
        return {
            'quality_score': round(quality_score, 3),
            'features_with_errors': features_with_errors,
            'issues': issues or ['No data quality issues detected'],
            'recommendations': recommendations
        }
    
    def _calculate_drift_severity(self, results: Dict[str, Any]) -> float:
        """Calculate drift severity score (0-1)"""
        severity_scores = []
        
        # Statistical drift severity
        if 'statistical' in results:
            stat_results = results['statistical']
            if not isinstance(stat_results, dict) or 'error' in stat_results:
                return 0.0
                
            mean_drift = stat_results.get('mean_drift', 0)
            std_drift = stat_results.get('std_drift', 0)
            p_value = stat_results.get('p_value_approx', 1.0)
            
            # Convert p-value to severity score
            p_score = max(0, 1 - p_value * 10)  # Lower p-value = higher severity
            drift_score = min(1, (mean_drift + std_drift) / 2)
            
            severity_scores.append(max(p_score, drift_score))
        
        # Distribution drift severity
        if 'distribution' in results:
            dist_results = results['distribution']
            if isinstance(dist_results, dict) and 'error' not in dist_results:
                if 'ks_test' in dist_results:
                    ks_p_value = dist_results['ks_test'].get('p_value', 1.0)
                    ks_score = max(0, 1 - ks_p_value * 10)
                    severity_scores.append(ks_score)
        
        return max(severity_scores) if severity_scores else 0.0
    
    def _get_status_description(self, status: str) -> str:
        """Get human-readable status description"""
        descriptions = {
            'stable': 'Data is stable with no significant drift detected',
            'minor_drift': 'Minor drift detected in some features - monitor closely',
            'warning': 'Moderate drift detected - consider model retraining',
            'critical': 'Severe drift detected - immediate action required',
            'error': 'Unable to analyze drift due to data issues'
        }
        return descriptions.get(status, 'Unknown status')
