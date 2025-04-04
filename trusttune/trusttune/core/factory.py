from typing import Literal, Union
from .base import BaseCalibrator
from .platt import PlattCalibrator
from .isotonic import IsotonicCalibrator
from .temperature import TemperatureCalibrator

CalibratorType = Literal["platt", "isotonic", "temperature"]

def create_calibrator(
    calibrator_type: CalibratorType
) -> BaseCalibrator:
    """Create a calibrator instance of the specified type.
    
    Args:
        calibrator_type: Type of calibrator to create
        
    Returns:
        An instance of the specified calibrator
        
    Raises:
        ValueError: If calibrator_type is not recognized
    """
    calibrators = {
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
        "temperature": TemperatureCalibrator
    }
    
    if calibrator_type not in calibrators:
        raise ValueError(
            f"Unknown calibrator type: {calibrator_type}. "
            f"Available types: {list(calibrators.keys())}"
        )
    
    return calibrators[calibrator_type]() 