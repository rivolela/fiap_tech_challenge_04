"""
Data Collector for Drift Monitoring
===================================

Handles collection and storage of reference and current data for drift analysis.
"""

import logging
import pandas as pd
from typing import Any, List, Dict, Optional
from datetime import datetime


class DataCollector:
    """Collects and manages data for drift analysis"""
    
    def __init__(self):
        self.reference_data = []
        self.current_data = []
        self.logger = logging.getLogger(__name__)
        self._reference_timestamp = None
        self._current_data_timestamps = []
    
    def collect_reference_data(self, data: Any, timestamp: Optional[datetime] = None):
        """
        Store reference/baseline data
        
        Args:
            data: Reference data (list, dict, or DataFrame)
            timestamp: When the reference data was collected
        """
        try:
            self.reference_data = data
            self._reference_timestamp = timestamp or datetime.now()
            self.logger.info(f"Reference data collected: {len(data)} samples")
            
        except Exception as e:
            self.logger.error(f"Error collecting reference data: {e}")
            raise
    
    def collect_current_data(self, data: Any, timestamp: Optional[datetime] = None):
        """
        Store current production data
        
        Args:
            data: Current data sample
            timestamp: When the data was collected
        """
        try:
            self.current_data.append(data)
            self._current_data_timestamps.append(timestamp or datetime.now())
            self.logger.info(f"Current data collected: {len(self.current_data)} total samples")
            
        except Exception as e:
            self.logger.error(f"Error collecting current data: {e}")
            raise
    
    def get_reference_data(self) -> pd.DataFrame:
        """Get reference data as DataFrame"""
        try:
            if not self.reference_data:
                return pd.DataFrame()
            
            if isinstance(self.reference_data, pd.DataFrame):
                return self.reference_data
            elif isinstance(self.reference_data, list):
                return pd.DataFrame(self.reference_data)
            elif isinstance(self.reference_data, dict):
                return pd.DataFrame([self.reference_data])
            else:
                return pd.DataFrame(self.reference_data)
                
        except Exception as e:
            self.logger.error(f"Error converting reference data to DataFrame: {e}")
            return pd.DataFrame()
    
    def get_current_data(self) -> pd.DataFrame:
        """Get current data as DataFrame"""
        try:
            if not self.current_data:
                return pd.DataFrame()
            
            return pd.DataFrame(self.current_data)
            
        except Exception as e:
            self.logger.error(f"Error converting current data to DataFrame: {e}")
            return pd.DataFrame()
    
    def get_recent_data(self, minutes: int = 60) -> pd.DataFrame:
        """
        Get current data from the last N minutes
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            DataFrame with recent data
        """
        try:
            if not self.current_data or not self._current_data_timestamps:
                return pd.DataFrame()
            
            cutoff_time = datetime.now() - pd.Timedelta(minutes=minutes)
            
            recent_data = []
            for data, timestamp in zip(self.current_data, self._current_data_timestamps):
                if timestamp >= cutoff_time:
                    recent_data.append(data)
            
            return pd.DataFrame(recent_data) if recent_data else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting recent data: {e}")
            return pd.DataFrame()
    
    def clear_current_data(self):
        """Clear accumulated current data"""
        self.current_data = []
        self._current_data_timestamps = []
        self.logger.info("Current data cleared")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        return {
            'reference_data': {
                'samples': len(self.reference_data) if self.reference_data else 0,
                'timestamp': self._reference_timestamp.isoformat() if self._reference_timestamp else None
            },
            'current_data': {
                'samples': len(self.current_data),
                'oldest_timestamp': self._current_data_timestamps[0].isoformat() if self._current_data_timestamps else None,
                'newest_timestamp': self._current_data_timestamps[-1].isoformat() if self._current_data_timestamps else None
            }
        }
