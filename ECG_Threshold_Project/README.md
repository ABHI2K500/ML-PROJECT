# ECG-Based Classification and Investigation of Decision Thresholds

## Team 03
1. Abhijith P
2. Noel Biju
3. Diya Krishna
4. Ann Mariya Biju

## Project Objective
Build a complete academic machine-learning project for ECG classification and investigate how changing decision thresholds affects the behaviour and performance of the model.

## Problem Statement
Default machine learning models typically use a 0.50 threshold for binary classification. In medical contexts, this may not be optimal depending on the cost of false positives vs false negatives.

## Research Question
How does changing the decision threshold affect the behaviour and performance of an ECG classification model?

## Dataset
We use the **PTB-XL** ECG dataset from PhysioNet.
The dataset must be placed in the `data/ptb-xl/` directory before running the code.

**Dataset Placement Instructions:**
1. Download the PTB-XL dataset from PhysioNet (https://physionet.org/content/ptb-xl/).
2. Extract the contents.
3. Place the `ptbxl_database.csv`, `scp_statements.csv`, and all `records100` folders directly into `ECG_Threshold_Project/data/ptb-xl/`.

## Installation & Environment Setup
Create a virtual environment and install the required dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\\Scripts\\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to train the model
Run the training pipeline from the root of the project:
```bash
python src/train_model.py
```
This will read the dataset, extract features, train the model, run the threshold experiment, generate graphs in `results/`, and save the model to `model/`.

## How to run the notebook
Start Jupyter Notebook and open the investigation notebook:
```bash
jupyter notebook notebooks/ECG_Threshold_Investigation.ipynb
```

## How to start Flask (Backend)
Run the Flask server:
```bash
python backend/app.py
```
The server will run at `http://localhost:5000`.

## How to open the frontend
Simply open `frontend/index.html` in your web browser. Or use a live server extension in VS Code.

## Expected input format
The web application expects an uploaded JSON file containing a 2D array of ECG signals representing `[timesteps, 12 leads]`. 
Example: A 500-timestep recording would be a JSON array of shape (500, 12).

## Output explanation
The web application displays:
- **Probability:** The probability assigned by the Logistic Regression model that the ECG pattern is abnormal.
- **Threshold:** The chosen decision threshold.
- **Classification:** Either "Normal ECG pattern" or "Abnormal ECG pattern".

## Project limitations
- This project uses simple statistical feature extraction rather than deep learning architectures which are state-of-the-art for ECG.
- For true medical AI validation, thresholds should be optimized on a validation set and tested on an unseen test set.

## Medical Disclaimer
**This application is an academic machine-learning demonstration and is not a medical diagnostic tool.**
