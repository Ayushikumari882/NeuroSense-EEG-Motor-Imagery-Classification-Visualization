# NeuroSense-EEG-Motor-Imagery-Classification-Visualization

NeuroSense is a full-stack EEG motor imagery classification system designed to identify left-hand versus right-hand imagined movements from EEG signals. The project integrates EEG preprocessing, feature extraction using Common Spatial Patterns (CSP), machine learning classification, and an interactive web dashboard for comprehensive analysis and visualization.

## Technologies Used

- **Backend:** Flask, MNE, NumPy, SciPy, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript, Plotly.js

## Supported Formats

The system supports:
- **EDF files**
- **MAT files**

## Outputs Provided

- Predicted class
- Confidence score
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix
- EEG waveform visualization
- Spectrogram
- Band power
- CSP pattern map
- Classifier benchmark comparison

## Key Features

- Load PhysioNet EEGBCI data through MNE
- Multi-format uploads (EDF, BDF, GDF, MAT, CSV, SET, FIF, ZIP bundles)
- 8-30 Hz preprocessing and epoch extraction
- CSP feature extraction
- Calibrated linear SVM classification
- Benchmarking against LDA
- Comprehensive validation metrics (accuracy, precision, recall, F1-score, ROC-AUC, cross-validation)
- Interactive visualizations: EEG waveform, spectrogram, band power, confusion matrix, classifier benchmark chart, CSP pattern heatmap
- Export results as CSV
- Save and reload report states
- Support for subject-based and multi-subject evaluations
- Premium dark clinical dashboard UI with Plotly visualizations

## Project Structure

```
NeuroSense/
├── app.py
├── config.json
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── saved_models/
├── .venv/
└── __pycache__/
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Ayushikumari882/NeuroSense-EEG-Motor-Imagery-Classification.git
   cd NeuroSense
   ```

2. Create and activate a virtual environment:
   - **Windows:**
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

If `python` does not work, try:
```
.venv\Scripts\python.exe app.py
```

## How It Works

The project follows this processing pipeline:

1. **Data Loading:** Load EEG datasets from EDF or MAT files.
2. **Preprocessing:** 
   - Channel standardization
   - 8–30 Hz bandpass filtering
   - Epoch extraction
3. **Feature Extraction:** Common Spatial Patterns (CSP)
4. **Classification:** SVM (primary) and LDA (benchmark)
5. **Evaluation:** Compute metrics like accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, cross-validation
6. **Output:** Interactive dashboard with visualizations

## Supported Input Formats

### EDF
Intended for raw EEG recordings containing:
- EEG channel signals
- Event annotations for motor imagery
- Left/right cue markers (e.g., T1 and T2)

### MAT
Supports MATLAB EEG structures such as:
- `imagery_left`, `imagery_right`, `srate`
- `data`, `labels`, `sfreq`
- `session.data`, `session.labels`, `session.sfreq`
- BNCI-like trial structures

## Algorithms Used

### Core Algorithms
- **Bandpass Filtering:** 8-30 Hz
- **Epoch Extraction**
- **Common Spatial Patterns (CSP):** For spatial feature extraction
- **Support Vector Machine (SVM):** Main classifier
- **Linear Discriminant Analysis (LDA):** Benchmark classifier
- **K-Fold Cross-Validation:** For robust evaluation

### Main Pipeline
- EEG preprocessing
- CSP feature extraction
- SVM classification
- LDA benchmarking
- Validation metrics computation
