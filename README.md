# 🔁 K-Fold Cross Validation — Model Comparison on Digits Dataset

A machine learning project that compares **Logistic Regression**, **SVM**, and **Random Forest** using **K-Fold Cross Validation** to find the best model for handwritten digit recognition.

## 📌 What It Does
- Loads the built-in **Digits dataset** from scikit-learn
- Trains and evaluates 3 models using a simple train/test split
- Applies **Stratified K-Fold Cross Validation** (3 splits) for robust evaluation
- Uses `cross_val_score` for quick multi-fold scoring
- Tunes Random Forest by testing different `n_estimators` (20, 40, 60)

## 📊 Model Comparison (cross_val_score)
| Model | Scores (5-Fold) |
|-------|----------------|
| Logistic Regression | ~0.922, 0.869, 0.941, 0.938, 0.896 |
| SVM | ~0.961, 0.944, 0.983, 0.988, 0.938 |
| Random Forest (40) | ~0.930, 0.894, 0.961, 0.963, 0.905 |

> ✅ **SVM performs best** with consistently high cross-validation scores.

## 🔧 Parameter Tuning
```python
cross_val_score(RandomForestClassifier(n_estimators=20), ...)  # baseline
cross_val_score(RandomForestClassifier(n_estimators=40), ...)  # better
cross_val_score(RandomForestClassifier(n_estimators=60), ...)  # optimal
```

## 🚀 Run It

**1. Install dependencies:**
```bash
pip install jupyter scikit-learn numpy
```

**2. Launch the notebook:**
```bash
jupyter notebook 11k_Fold_Cross_Validation.ipynb
```

## 🛠️ Built With
Python · NumPy · Scikit-learn · Jupyter Notebook
