"""Edge case tests for calibrators."""
import pytest
import numpy as np
from trusttune.core.factory import create_calibrator

class TestCoreEdgeCases:
    @pytest.fixture(params=["platt", "isotonic", "temperature"])
    def calibrator_type(self, request):
        return request.param

    def test_empty_arrays(self, calibrator_type):
        """Test behavior with empty arrays."""
        calibrator = create_calibrator(calibrator_type)
        with pytest.raises(ValueError, match="Empty arrays are not allowed"):
            calibrator._validate_inputs(np.array([]))

    def test_single_class(self, calibrator_type):
        """Test behavior with single class data."""
        calibrator = create_calibrator(calibrator_type)
        with pytest.raises(ValueError, match="At least two classes required"):
            calibrator._validate_inputs(np.array([0.5, 0.6]), np.array([1, 1]))

    def test_extreme_values(self, calibrator_type):
        """Test behavior with extreme values."""
        calibrator = create_calibrator(calibrator_type)
        with pytest.raises(ValueError, match="Scores must be between 0 and 1"):
            calibrator._validate_inputs(np.array([1.5]))

    def test_shape_mismatch(self, calibrator_type):
        """Test behavior with mismatched shapes."""
        calibrator = create_calibrator(calibrator_type)
        with pytest.raises(ValueError, match="Number of scores and labels must match"):
            calibrator._validate_inputs(np.array([0.5, 0.6]), np.array([1]))

    def test_non_binary_labels(self, calibrator_type):
        """Test behavior with non-binary labels."""
        calibrator = create_calibrator(calibrator_type)
        with pytest.raises(ValueError, match="Labels must be binary"):
            calibrator._validate_inputs(np.array([0.5, 0.6]), np.array([1, 2])) 