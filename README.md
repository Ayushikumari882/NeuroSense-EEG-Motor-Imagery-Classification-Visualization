# NeuroSense - EEG Motor Imagery Classification

Standalone Flask dashboard for EEG motor imagery analysis with:

- PhysioNet EEGBCI loading through MNE
- Multi-format uploads for EDF, BDF, GDF, MAT, CSV, SET, FIF, and ZIP bundles
- 8-30 Hz preprocessing and epoch extraction
- CSP feature extraction
- Calibrated linear SVM classification
- Premium dark clinical dashboard UI with Plotly visualizations

## Run

```powershell
.\.venv\Scripts\activate
python app.py
```

Open `http://127.0.0.1:5000`

## Project Files

- `app.py`
- `templates/index.html`
- `static/style.css`
- `static/script.js`
- `requirements.txt`
