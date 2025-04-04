class TrustTuneError(Exception):
    """Base exception for TrustTune."""
    pass

class CalibrationError(TrustTuneError):
    """Raised when there's an error during calibration."""
    pass

class ValidationError(TrustTuneError):
    """Raised when input validation fails."""
    pass

class StateError(TrustTuneError):
    """Raised when there's an error with calibrator state."""
    pass

class DriftDetectionError(TrustTuneError):
    """Raised when there's an error in drift detection."""
    pass 