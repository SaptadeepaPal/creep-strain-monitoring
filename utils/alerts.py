"""
alerts.py
=========
Alert logic for creep strain monitoring system
Stateless functions for threshold and trend-based alerts
"""

import numpy as np
from typing import List, Tuple


def threshold_alert_check(predicted_strain: float, warning_threshold: float) -> str:
    """
    Determine alert level based on absolute strain threshold.
    
    This is the primary safety mechanism - alerts if strain exceeds
    known dangerous levels regardless of historical trends.
    
    Args:
        predicted_strain: Current predicted strain percentage
        warning_threshold: Strain threshold that triggers warnings
        
    Returns:
        str: "WARNING" if strain >= threshold, "NORMAL" otherwise
    """
    if predicted_strain >= warning_threshold:
        return "WARNING"
    else:
        return "NORMAL"


def trend_alert_check(strain_history: List[float]) -> str:
    """
    Detect abnormal strain increases based on recent prediction history.
    
    Early warning system: Even if strain is below thresholds, a sudden
    increase could indicate accelerating creep (dangerous).
    
    Logic:
    - Requires at least 5 historical points for analysis
    - Computes linear regression slope over history
    - Positive slope indicates increasing strain trend
    
    Args:
        strain_history: List of recent strain predictions (max 5)
        
    Returns:
        str: "INSUFFICIENT_DATA" (< 5 points)
             "INCREASING" (positive slope)
             "STABLE" (zero or negative slope)
    """
    # Need minimum history for trend analysis
    if len(strain_history) < 5:
        return "INSUFFICIENT_DATA"
    
    # Prepare data for linear regression
    x = np.arange(len(strain_history))  # Time indices
    y = np.array(strain_history)        # Strain values
    
    # Compute linear fit (degree 1 polynomial)
    # Returns [slope, intercept]
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    
    # Positive slope indicates increasing strain
    if slope > 0:
        return "INCREASING"
    else:
        return "STABLE"


def fuse_alerts(threshold_status: str, trend_status: str) -> str:
    """
    Combine threshold-based and trend-based alerts into final decision.
    
    Priority hierarchy:
    1. CRITICAL ALERT - Both threshold warning AND increasing trend
    2. WARNING ALERT - Threshold warning only
    3. TREND ALERT - Increasing trend only
    4. NORMAL - No alerts
    
    Args:
        threshold_status: Result from threshold_alert_check
        trend_status: Result from trend_alert_check
        
    Returns:
        str: Final fused alert level
    """
    # Both threshold and trend warnings active (highest severity)
    if threshold_status == "WARNING" and trend_status == "INCREASING":
        return "CRITICAL ALERT"
    
    # Threshold warning only
    elif threshold_status == "WARNING":
        return "WARNING ALERT"
    
    # Trend warning only (early indicator)
    elif trend_status == "INCREASING":
        return "TREND ALERT"
    
    # No alerts
    else:
        return "NORMAL"


def get_operator_guidance(final_alert: str) -> str:
    """
    Provide human-readable guidance based on alert severity.
    
    Args:
        final_alert: Final fused alert decision
        
    Returns:
        str: Operator guidance message
    """
    guidance_map = {
        "NORMAL": "System operating within safe limits. Continue routine monitoring.",
        "TREND ALERT": "Monitor closely for creep acceleration. Review recent data trends.",
        "WARNING ALERT": "Inspection recommended. Evaluate component condition.",
        "CRITICAL ALERT": "Immediate action required. Initiate emergency protocols."
    }
    
    return guidance_map.get(final_alert, "Unknown alert status. Contact system administrator.")


def get_alert_color(final_alert: str) -> str:
    """
    Get color code for alert visualization in UI.
    
    Args:
        final_alert: Final fused alert decision
        
    Returns:
        str: Color name compatible with Streamlit
    """
    color_map = {
        "NORMAL": "green",
        "TREND ALERT": "blue",
        "WARNING ALERT": "orange",
        "CRITICAL ALERT": "red"
    }
    
    return color_map.get(final_alert, "gray")


def get_alert_severity(final_alert: str) -> int:
    """
    Get numeric severity level for alert sorting/filtering.
    
    Args:
        final_alert: Final fused alert decision
        
    Returns:
        int: Severity level (1=lowest, 4=highest)
    """
    severity_map = {
        "NORMAL": 1,
        "TREND ALERT": 2,
        "WARNING ALERT": 3,
        "CRITICAL ALERT": 4
    }
    
    return severity_map.get(final_alert, 0)