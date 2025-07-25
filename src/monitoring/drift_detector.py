"""
Drift Detection Engine
=====================

Implements various statistical methods to detect data drift.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class DriftDetector:
    """Detects drift between datasets using statistical methods"""
    
    def __init__(self, default_threshold: float = 0.05):
        self.default_threshold = default_threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_drift(
        self, 
        reference_data: Any, 
        current_data: Any, 
        threshold: Optional[float] = None,
        methods: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data
        
        Args:
            reference_data: Baseline data
            current_data: Current data to compare
            threshold: P-value threshold for significance
            methods: List of detection methods to use
            
        Returns:
            Dict with drift detection results
        """
        threshold = threshold or self.default_threshold
        methods = methods or ['statistical', 'distribution']
        
        try:
            if self._is_empty_data(reference_data) or self._is_empty_data(current_data):
                return {'error': 'Empty datasets provided'}
            
            # Convert to DataFrames
            ref_df = self._to_dataframe(reference_data)
            curr_df = self._to_dataframe(current_data)
            
            if ref_df.empty or curr_df.empty:
                return {'error': 'Could not convert data to DataFrame'}
            
            drift_results = {}
            common_columns = set(ref_df.columns) & set(curr_df.columns)
            
            if not common_columns:
                return {'error': 'No common columns found between datasets'}
            
            # Analyze each common column
            for column in common_columns:
                try:
                    column_results = {}
                    
                    if 'statistical' in methods:
                        column_results['statistical'] = self._detect_statistical_drift(
                            ref_df[column], curr_df[column], threshold
                        )
                    
                    if 'distribution' in methods:
                        column_results['distribution'] = self._detect_distribution_drift(
                            ref_df[column], curr_df[column], threshold
                        )
                    
                    # Overall drift decision for this column
                    column_results['drift_detected'] = self._determine_overall_drift(column_results)
                    
                    drift_results[column] = column_results
                    
                except Exception as e:
                    drift_results[column] = {'error': str(e)}
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error in drift detection: {e}")
            return {'error': str(e)}
    
    def _detect_statistical_drift(
        self, 
        ref_series: pd.Series, 
        curr_series: pd.Series, 
        threshold: float
    ) -> Dict[str, Any]:
        """Detect drift using basic statistical measures"""
        try:
            # Clean data
            ref_clean = ref_series.dropna()
            curr_clean = curr_series.dropna()
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return {'error': 'No valid data after cleaning'}
            
            # Calculate statistics
            ref_mean = float(ref_clean.mean())
            curr_mean = float(curr_clean.mean())
            ref_std = float(ref_clean.std())
            curr_std = float(curr_clean.std())
            ref_median = float(ref_clean.median())
            curr_median = float(curr_clean.median())
            
            # Calculate drift metrics
            mean_drift = abs(curr_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else 0
            std_drift = abs(curr_std - ref_std) / abs(ref_std) if ref_std != 0 else 0
            median_drift = abs(curr_median - ref_median) / abs(ref_median) if ref_median != 0 else 0
            
            # Statistical significance test (simple t-test alternative)
            pooled_std = np.sqrt(((len(ref_clean) - 1) * ref_std**2 + (len(curr_clean) - 1) * curr_std**2) / 
                                (len(ref_clean) + len(curr_clean) - 2))
            
            if pooled_std > 0:
                t_stat = abs(curr_mean - ref_mean) / (pooled_std * np.sqrt(1/len(ref_clean) + 1/len(curr_clean)))
                # Approximate p-value (rough estimate)
                p_value_approx = 2 * (1 - self._t_cdf(abs(t_stat), len(ref_clean) + len(curr_clean) - 2))
            else:
                t_stat = 0
                p_value_approx = 1.0
            
            return {
                'mean_drift': float(mean_drift),
                'std_drift': float(std_drift),
                'median_drift': float(median_drift),
                't_statistic': float(t_stat),
                'p_value_approx': float(p_value_approx),
                'drift_detected': mean_drift > 0.1 or std_drift > 0.2 or p_value_approx < threshold,
                'statistics': {
                    'reference': {
                        'mean': ref_mean,
                        'std': ref_std,
                        'median': ref_median,
                        'count': len(ref_clean)
                    },
                    'current': {
                        'mean': curr_mean,
                        'std': curr_std,
                        'median': curr_median,
                        'count': len(curr_clean)
                    }
                }
            }
            
        except Exception as e:
            return {'error': f'Statistical drift detection failed: {str(e)}'}
    
    def _detect_distribution_drift(
        self, 
        ref_series: pd.Series, 
        curr_series: pd.Series, 
        threshold: float
    ) -> Dict[str, Any]:
        """Detect drift using distribution comparison"""
        try:
            # Try to use scipy if available
            try:
                from scipy.stats import ks_2samp, chi2_contingency
                scipy_available = True
            except ImportError:
                scipy_available = False
            
            ref_clean = ref_series.dropna()
            curr_clean = curr_series.dropna()
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return {'error': 'No valid data after cleaning'}
            
            result = {}
            
            if scipy_available:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = ks_2samp(ref_clean, curr_clean)
                result['ks_test'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_pvalue),
                    'drift_detected': ks_pvalue < threshold
                }
                
                # For categorical data, try chi-square test
                if self._is_categorical(ref_series):
                    try:
                        ref_counts = ref_clean.value_counts()
                        curr_counts = curr_clean.value_counts()
                        
                        # Align the counts
                        all_categories = set(ref_counts.index) | set(curr_counts.index)
                        ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                        curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                        
                        if len(all_categories) > 1:
                            chi2_stat, chi2_pvalue, _, _ = chi2_contingency([ref_aligned, curr_aligned])
                            result['chi2_test'] = {
                                'statistic': float(chi2_stat),
                                'p_value': float(chi2_pvalue),
                                'drift_detected': chi2_pvalue < threshold
                            }
                    except Exception:
                        pass
            else:
                # Fallback: simple binning comparison
                result = self._simple_distribution_comparison(ref_clean, curr_clean, threshold)
            
            # Overall drift decision
            drift_detected = False
            if 'ks_test' in result:
                drift_detected = drift_detected or result['ks_test']['drift_detected']
            if 'chi2_test' in result:
                drift_detected = drift_detected or result['chi2_test']['drift_detected']
            if 'simple_comparison' in result:
                drift_detected = drift_detected or result['simple_comparison']['drift_detected']
            
            result['drift_detected'] = drift_detected
            return result
            
        except Exception as e:
            return {'error': f'Distribution drift detection failed: {str(e)}'}
    
    def _simple_distribution_comparison(
        self, 
        ref_series: pd.Series, 
        curr_series: pd.Series, 
        threshold: float
    ) -> Dict[str, Any]:
        """Simple distribution comparison when scipy is not available"""
        try:
            # Create bins based on reference data
            n_bins = min(10, len(ref_series.unique()))
            bins = pd.cut(ref_series, bins=n_bins, duplicates='drop')
            
            # Count occurrences in each bin
            ref_counts = bins.value_counts(normalize=True).sort_index()
            curr_bins = pd.cut(curr_series, bins=bins.cat.categories, duplicates='drop')
            curr_counts = curr_bins.value_counts(normalize=True).sort_index()
            
            # Calculate simple difference metric
            diff_sum = sum(abs(ref_counts.get(bin_val, 0) - curr_counts.get(bin_val, 0)) 
                          for bin_val in ref_counts.index)
            
            return {
                'simple_comparison': {
                    'difference_sum': float(diff_sum),
                    'drift_detected': diff_sum > 0.2,  # Simple threshold
                    'note': 'Simple binning comparison (scipy not available)'
                }
            }
            
        except Exception as e:
            return {'error': f'Simple distribution comparison failed: {str(e)}'}
    
    def _determine_overall_drift(self, column_results: Dict[str, Any]) -> bool:
        """Determine overall drift status for a column"""
        drift_indicators = []
        
        if 'statistical' in column_results:
            drift_indicators.append(column_results['statistical'].get('drift_detected', False))
        
        if 'distribution' in column_results:
            drift_indicators.append(column_results['distribution'].get('drift_detected', False))
        
        # Return True if any method detected drift
        return any(drift_indicators)
    
    def _is_empty_data(self, data: Any) -> bool:
        """Check if data is empty"""
        if data is None:
            return True
        if hasattr(data, '__len__'):
            return len(data) == 0
        return False
    
    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert data to DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (list, dict)):
            return pd.DataFrame(data)
        else:
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame()
    
    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if series contains categorical data"""
        if series.dtype == 'object':
            return True
        if hasattr(series, 'cat'):
            return True
        # Check if numeric data has few unique values (might be categorical)
        if series.dtype in ['int64', 'float64']:
            unique_ratio = len(series.unique()) / len(series)
            return unique_ratio < 0.1  # Less than 10% unique values
        return False
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF (very rough approximation)"""
        # This is a very rough approximation for when scipy is not available
        # In practice, you'd want to use scipy.stats.t.cdf
        if df > 30:
            # For large df, t-distribution approaches normal
            return 0.5 * (1 + np.sign(t) * np.sqrt(1 - np.exp(-2 * t**2 / np.pi)))
        else:
            # Very rough approximation
            return 0.5 + 0.5 * np.sign(t) * min(1, abs(t) / 3)