"""Calibrator factory implementation."""
from typing import Dict, Type
from .base import BaseCalibrator
from .platt import PlattCalibrator
from .isotonic import IsotonicCalibrator
from .temperature import TemperatureCalibrator

CALIBRATOR_TYPES: Dict[str, Type[BaseCalibrator]] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator
}

def create_calibrator(calibrator_type: str) -> BaseCalibrator:
    """Create a calibrator instance.
    
    Args:
        calibrator_type: Type of calibrator to create
        
    Returns:
        Calibrator instance
        
    Raises:
        ValueError: If calibrator_type is not supported
    """
    if calibrator_type not in CALIBRATOR_TYPES:
        raise ValueError(
            f"Unknown calibrator type: {calibrator_type}. "
            f"Supported types are: {list(CALIBRATOR_TYPES.keys())}"
        )
    
    return CALIBRATOR_TYPES[calibrator_type]() 