// TrustTune UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const calibrationForm = document.getElementById('calibrationForm');
    const fileInput = document.querySelector('input[type="file"]');
    const resultsContainer = document.getElementById('resultsContainer');
    const calibrationPlot = document.getElementById('calibrationPlot');
    const loadingIndicator = document.getElementById('loadingIndicator');
    
    // Event listeners
    if (calibrationForm) {
        calibrationForm.addEventListener('submit', handleCalibrationSubmit);
    }
    
    // Handle form submission
    async function handleCalibrationSubmit(event) {
        event.preventDefault();
        
        if (!fileInput.files.length) {
            showMessage('Please select a CSV file', 'error');
            return;
        }
        
        showLoading(true);
        
        try {
            const formData = new FormData(calibrationForm);
            const calibratorType = formData.get('calibrator_type');
            
            // Parse CSV file
            const file = fileInput.files[0];
            const data = await parseCSV(file);
            
            if (!data || !data.scores || !data.labels) {
                showMessage('Invalid CSV format. Please ensure it contains "score" and "label" columns.', 'error');
                showLoading(false);
                return;
            }
            
            // Call API
            const response = await fetch('/calibrate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    scores: data.scores,
                    labels: data.labels,
                    calibrator_type: calibratorType
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Calibration failed');
            }
            
            const result = await response.json();
            
            // Display results
            displayResults(result, data);
            
            // Create reliability diagram
            createReliabilityDiagram(data.scores, result.calibrated_scores, data.labels);
            
        } catch (error) {
            console.error('Error:', error);
            showMessage(error.message || 'An error occurred during calibration', 'error');
        } finally {
            showLoading(false);
        }
    }
    
    // Parse CSV file
    async function parseCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = function(event) {
                const csvData = event.target.result;
                const lines = csvData.split('\n');
                const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
                
                const scoreIndex = headers.indexOf('score');
                const labelIndex = headers.indexOf('label');
                
                if (scoreIndex === -1 || labelIndex === -1) {
                    reject(new Error('CSV must contain "score" and "label" columns'));
                    return;
                }
                
                const scores = [];
                const labels = [];
                
                for (let i = 1; i < lines.length; i++) {
                    if (!lines[i].trim()) continue;
                    
                    const values = lines[i].split(',');
                    const score = parseFloat(values[scoreIndex]);
                    const label = parseInt(values[labelIndex], 10);
                    
                    if (!isNaN(score) && !isNaN(label)) {
                        scores.push(score);
                        labels.push(label);
                    }
                }
                
                resolve({ scores, labels });
            };
            
            reader.onerror = function() {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsText(file);
        });
    }
    
    // Display calibration results
    function displayResults(result, originalData) {
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('d-none');
        
        // Create metrics section
        const metricsCard = document.createElement('div');
        metricsCard.className = 'card mb-4';
        metricsCard.innerHTML = `
            <div class="card-header">
                <h5 class="mb-0">Calibration Metrics</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Expected Calibration Error:</strong> ${result.metrics.ece.toFixed(4)}</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Brier Score:</strong> ${result.metrics.brier_score.toFixed(4)}</p>
                    </div>
                </div>
            </div>
        `;
        
        // Create data preview section
        const dataPreview = document.createElement('div');
        dataPreview.className = 'card';
        
        // Create table with original and calibrated scores
        let tableRows = '';
        const maxRows = Math.min(10, result.calibrated_scores.length);
        
        for (let i = 0; i < maxRows; i++) {
            tableRows += `
                <tr>
                    <td>${i + 1}</td>
                    <td>${originalData.scores[i].toFixed(4)}</td>
                    <td>${result.calibrated_scores[i].toFixed(4)}</td>
                    <td>${originalData.labels[i]}</td>
                </tr>
            `;
        }
        
        dataPreview.innerHTML = `
            <div class="card-header">
                <h5 class="mb-0">Data Preview (First ${maxRows} rows)</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Original Score</th>
                                <th>Calibrated Score</th>
                                <th>True Label</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${tableRows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        resultsContainer.appendChild(metricsCard);
        resultsContainer.appendChild(dataPreview);
    }
    
    // Create reliability diagram
    function createReliabilityDiagram(originalScores, calibratedScores, labels) {
        if (!calibrationPlot) return;
        
        // Create bins for reliability diagram
        const numBins = 10;
        const binSize = 1.0 / numBins;
        
        // Function to calculate bin statistics
        function calculateBinStats(scores, labels) {
            const bins = Array(numBins).fill().map(() => ({ count: 0, positive: 0 }));
            
            scores.forEach((score, i) => {
                const binIndex = Math.min(Math.floor(score / binSize), numBins - 1);
                bins[binIndex].count++;
                bins[binIndex].positive += labels[i];
            });
            
            // Calculate fraction of positives and confidence for each bin
            return bins.map((bin, i) => {
                const confidence = (i + 0.5) * binSize;
                const accuracy = bin.count > 0 ? bin.positive / bin.count : 0;
                return { confidence, accuracy, count: bin.count };
            });
        }
        
        const originalBins = calculateBinStats(originalScores, labels);
        const calibratedBins = calculateBinStats(calibratedScores, labels);
        
        // Filter out empty bins
        const filteredOriginalBins = originalBins.filter(bin => bin.count > 0);
        const filteredCalibratedBins = calibratedBins.filter(bin => bin.count > 0);
        
        // Create perfect calibration line
        const perfectX = [0, 1];
        const perfectY = [0, 1];
        
        // Create plot
        const data = [
            {
                x: filteredOriginalBins.map(bin => bin.confidence),
                y: filteredOriginalBins.map(bin => bin.accuracy),
                mode: 'markers+lines',
                name: 'Original',
                marker: { size: 8, color: 'blue' }
            },
            {
                x: filteredCalibratedBins.map(bin => bin.confidence),
                y: filteredCalibratedBins.map(bin => bin.accuracy),
                mode: 'markers+lines',
                name: 'Calibrated',
                marker: { size: 8, color: 'green' }
            },
            {
                x: perfectX,
                y: perfectY,
                mode: 'lines',
                name: 'Perfect Calibration',
                line: { dash: 'dash', color: 'gray' }
            }
        ];
        
        const layout = {
            title: 'Reliability Diagram',
            xaxis: { title: 'Confidence', range: [0, 1] },
            yaxis: { title: 'Accuracy', range: [0, 1] },
            legend: { orientation: 'h', y: -0.2 },
            width: 500,
            height: 400,
            margin: { l: 50, r: 50, t: 50, b: 50 }
        };
        
        Plotly.newPlot(calibrationPlot, data, layout);
    }
    
    // Show/hide loading indicator
    function showLoading(isLoading) {
        if (loadingIndicator) {
            loadingIndicator.style.display = isLoading ? 'block' : 'none';
        }
    }
    
    // Show message
    function showMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                alertDiv.classList.remove('show');
                setTimeout(() => alertDiv.remove(), 150);
            }, 5000);
        }
    }
});
