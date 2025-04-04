"""
End-to-end tests for the TrustTune UI
"""
import pytest
import os
import time
from playwright.sync_api import Page, expect

# Base URL for tests
BASE_URL = "http://localhost:8000"

def test_home_page_loads(page: Page):
    """Test that the home page loads correctly."""
    # Navigate to the home page
    page.goto(BASE_URL)
    
    # Check that the title is correct
    expect(page).to_have_title("TrustTune - ML Score Calibration")
    
    # Check that the main elements are present
    expect(page.locator("h1")).to_contain_text("TrustTune")
    expect(page.locator("#calibrationForm")).to_be_visible()
    expect(page.locator("#calibrationPlot")).to_be_visible()

def test_calibration_workflow(page: Page):
    """Test the end-to-end calibration workflow."""
    # Navigate to the home page
    page.goto(BASE_URL)
    
    # Select calibrator type
    page.select_option("select[name='calibrator_type']", "platt")
    
    # Upload sample data
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "static", "sample_data.csv")
    page.locator("input[type='file']").set_input_files(sample_path)
    
    # Submit the form
    page.click("button[type='submit']")
    
    # Wait for results to appear
    page.wait_for_selector("#resultsContainer:not(.d-none)", timeout=10000)
    
    # Check that the results are displayed
    expect(page.locator("#resultsContainer")).to_be_visible()
    expect(page.locator("#calibrationPlot")).to_be_visible()
    
    # Check that the metrics are displayed
    expect(page.locator("text=Expected Calibration Error")).to_be_visible()
    expect(page.locator("text=Brier Score")).to_be_visible()

def test_sample_data_download(page: Page):
    """Test that the sample data can be downloaded."""
    # Navigate to the home page
    page.goto(BASE_URL)
    
    # Check that the sample data link is present
    expect(page.locator("text=Download Sample CSV")).to_be_visible()
    
    # Click the download link and wait for download to start
    with page.expect_download() as download_info:
        page.click("a[download]")
    
    download = download_info.value
    
    # Check that the file was downloaded
    assert download.suggested_filename == "sample_data.csv"
    
    # Save the file to verify its contents
    download_path = os.path.join(os.path.dirname(__file__), "downloaded_sample.csv")
    download.save_as(download_path)
    
    # Check that the file exists
    assert os.path.exists(download_path)
    
    # Clean up
    os.remove(download_path)
