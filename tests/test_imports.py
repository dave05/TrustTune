def test_package_imports():
    """Test that all package modules can be imported."""
    import trusttune
    import trusttune.core
    import trusttune.api
    import trusttune.monitoring
    import trusttune.streaming
    import trusttune.utils
    
    assert trusttune.__version__ == "0.1.0" 