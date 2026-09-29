const fileInput = document.getElementById('ecgFileInput');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const errorContainer = document.getElementById('errorContainer');
const previewContainer = document.getElementById('previewContainer');
const ctx = document.getElementById('ecgChart').getContext('2d');

let ecgData = null;
let chartInstance = null;

// Handle slider change
thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Handle file upload
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        try {
            const data = JSON.parse(event.target.result);
            // Basic validation
            if (!Array.isArray(data)) {
                throw new Error("Invalid format: Expected a JSON array.");
            }
            ecgData = data;
            analyzeBtn.disabled = false;
            hideError();
            plotPreview(data);
        } catch (err) {
            showError("Could not parse JSON file. Ensure it contains a valid ECG array.");
            ecgData = null;
            analyzeBtn.disabled = true;
            previewContainer.classList.add('hidden');
        }
    };
    reader.onerror = () => showError("Error reading file.");
    reader.readAsText(file);
});

// Analyze Button Click
analyzeBtn.addEventListener('click', async () => {
    if (!ecgData) return;
    
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'ANALYZING...';
    hideError();
    resultsSection.classList.add('hidden');
    
    const threshold = parseFloat(thresholdSlider.value);
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ecg_signal: ecgData,
                threshold: threshold
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `Server error: ${response.status}`);
        }
        
        displayResults(result);
    } catch (err) {
        showError(`Analysis failed: ${err.message}`);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE ECG';
    }
});

function displayResults(data) {
    document.getElementById('resultProb').textContent = `${(data.probability * 100).toFixed(1)}%`;
    document.getElementById('resultThreshold').textContent = `${(data.threshold * 100).toFixed(0)}%`;
    
    const classEl = document.getElementById('resultClass');
    classEl.textContent = data.prediction;
    
    classEl.className = 'result-value result-badge'; // reset
    if (data.prediction.toLowerCase().includes('abnormal')) {
        classEl.classList.add('abnormal');
    } else {
        classEl.classList.add('normal');
    }
    
    resultsSection.classList.remove('hidden');
}

function showError(msg) {
    errorContainer.textContent = msg;
    errorContainer.classList.remove('hidden');
}

function hideError() {
    errorContainer.classList.add('hidden');
}

function plotPreview(data) {
    // Assuming data is shape [timesteps, 12] or similar. We plot the first column (Lead I)
    // Just plot the first 500 samples
    let lead1 = [];
    if (Array.isArray(data[0])) {
        lead1 = data.slice(0, 500).map(row => row[0]);
    } else {
        // 1D array fallback
        lead1 = data.slice(0, 500);
    }
    
    const labels = Array.from({length: lead1.length}, (_, i) => i);
    
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    previewContainer.classList.remove('hidden');
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'ECG Signal (Lead I)',
                data: lead1,
                borderColor: '#ef4444',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: { display: false },
                y: { display: true }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
