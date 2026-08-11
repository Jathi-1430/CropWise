# CropWise 2.0

An upgraded **Explainable Crop Classifier + Farm Decision Support System** built with Streamlit and scikit-learn.

## Features
- Crop classification using Random Forest
- Top 3 crop predictions
- Model confidence scores
- "Why this crop?" explanation using learned crop profiles
- Date -> season display
- What-if lab for exploring a target crop
- Model accuracy and feature importance dashboard
- Login screen for demo use
- One-click Windows launcher

## Dataset
Put your crop recommendation CSV in the project root with this filename:

`Crop_recommendation.csv`

Required columns:
`N, P, K, temperature, humidity, ph, rainfall, label`

The code also accepts a `crop` target column.

## Run on Windows
Double-click:

`START_CROPWISE.bat`

Or use:
```bash
python -m streamlit run app.py
```

Demo login:
- User ID: `farmer`
- Password: `cropwise`

## Important
The "confidence" is a model probability estimate, not a guarantee of crop yield.
The What-if Lab compares inputs with training-data patterns; it is not a fertilizer prescription.
For real farming decisions, validate with local soil tests, weather information, and agricultural experts.

Streamlit supports local `streamlit run app.py` workflows and Community Cloud deployment with a GitHub repository and requirements file.
