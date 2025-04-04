import pytest
from trusttune.core.factory import create_calibrator
from trusttune.core.platt import PlattCalibrator
from trusttune.core.isotonic import IsotonicCalibrator
from trusttune.core.temperature import TemperatureCalibrator


def test_create_calibrator():
    # Test creating each type of calibrator
    platt = create_calibrator("platt")
    assert isinstance(platt, PlattCalibrator)
    
    isotonic = create_calibrator("isotonic")
    assert isinstance(isotonic, IsotonicCalibrator)
    
    temperature = create_calibrator("temperature")
    assert isinstance(temperature, TemperatureCalibrator)


def test_create_calibrator_invalid_type():
    with pytest.raises(ValueError, match="Unknown calibrator type"):
        create_calibrator("invalid") 