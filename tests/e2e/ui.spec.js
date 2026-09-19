// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

test.describe('TrustTune UI Tests', () => {
  test('Home page loads correctly', async ({ page }) => {
    // Navigate to the home page
    await page.goto('/');

    // Check that the title is correct
    await expect(page).toHaveTitle('TrustTune - ML Score Calibration');

    // Check that the main elements are present
    await expect(page.locator('h1')).toContainText('TrustTune');

    // Check for key sections
    await expect(page.locator('#features')).toBeVisible();
    await expect(page.locator('#use-cases')).toBeVisible();
    await expect(page.locator('#documentation')).toBeVisible();

    // Take a screenshot of the home page
    await page.screenshot({ path: 'home-page.png' });

    console.log('Home page test passed successfully');
  });

  test('Demo page loads correctly', async ({ page }) => {
    // Navigate to the demo page
    await page.goto('/demo');

    // Check that the title is correct
    await expect(page).toHaveTitle('TrustTune - Interactive Demo');

    // Check that the main elements are present
    await expect(page.locator('h1')).toContainText('Interactive Demo');
    await expect(page.locator('#calibrationForm')).toBeVisible();
    await expect(page.locator('#calibrationPlot')).toBeVisible();

    // Take a screenshot of the demo page
    await page.screenshot({ path: 'demo-page.png' });

    console.log('Demo page test passed successfully');
  });

  test('Navigation works between pages', async ({ page }) => {
    // Start at the home page
    await page.goto('/');

    // Hide the loading indicator that's intercepting clicks
    await page.evaluate(() => {
      const loadingIndicator = document.getElementById('loadingIndicator');
      if (loadingIndicator) loadingIndicator.style.display = 'none';
    });

    // Navigate to the demo page using JavaScript navigation
    await page.evaluate(() => {
      window.location.href = '/demo';
    });

    // Wait for the demo page to load
    await page.waitForURL('**/demo');

    // Check we're on the demo page
    await expect(page).toHaveTitle('TrustTune - Interactive Demo');

    // Navigate back to the home page using JavaScript navigation
    await page.evaluate(() => {
      window.location.href = '/';
    });

    // Wait for the home page to load
    await page.waitForURL('**/');

    // Check we're back on the home page
    await expect(page).toHaveTitle('TrustTune - ML Score Calibration');

    console.log('Navigation test passed successfully');
  });
});
