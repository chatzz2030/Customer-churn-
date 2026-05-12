# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer will churn (leave the service), built on the **IBM Telco Customer Churn** dataset. Includes a full EDA pipeline, two trained classifiers, and an interactive **Streamlit web app**.

---



## 📌 Problem Statement

Telecom companies lose millions each year due to customer churn. This project builds a **binary classifier** to identify customers at risk of churning, enabling proactive retention strategies.

- **Target variable:** `Churn` — whether a customer left within the last month (`Yes` / `No`)
- **Dataset:** IBM Telco Customer Churn — 7,032 customers, 21 features

---

## 🔬 Workflow Summary

### 1. Data Loading & Exploration
- Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Inspected head, shape, dtypes
- `TotalCharges` was stored as `object` → converted to `float64`
- Dropped 11 rows with missing `TotalCharges` → final shape: **(7032, 21)**

### 2. Exploratory Data Analysis (EDA)
| Visualization | Key Insight |
|---|---|
| Pie chart — Churn rate | ~26.5% of customers churned |
| Bar chart — Contract vs Churn | Month-to-month customers churn most |
| Histogram — Tenure vs Churn | New customers (< 6 months) churn more |
| KDE — Monthly Charges vs Churn | Higher charges → higher churn |

### 3. Data Preprocessing
1. Dropped `customerID` (non-informative)
2. Encoded `Churn`: `Yes → 1`, `No → 0`
3. Applied **one-hot encoding** (`pd.get_dummies`, `drop_first=True`) to all object columns
4. Applied **StandardScaler** to numerical columns: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
5. Final feature matrix shape: **(7032, 30)**

### 4. Train / Test Split
- 80% train / 20% test
- `stratify=y` to preserve class balance
- `random_state=42`

### 5. Model Training & Evaluation

#### Logistic Regression ✅ *(saved as `model.pkl`)*
| Metric | Score |
|---|---|
| Accuracy | **80.4%** |
| Precision (Churn) | 65% |
| Recall (Churn) | 57% |
| F1-Score (Churn) | 0.61 |

#### Random Forest Classifier
| Metric | Score |
|---|---|
| Accuracy | 79.0% |
| Precision (Churn) | 63% |
| Recall (Churn) | 52% |
| F1-Score (Churn) | 0.57 |

> Logistic Regression was chosen as the production model for its slightly better performance and interpretability.

### 6. Model Serialization
```python
import joblib
joblib.dump(model, 'model.pkl')
```

---

## 🚀 Running the Streamlit App

### Prerequisites
```bash
pip install streamlit scikit-learn pandas numpy
```

### Launch
```bash
cd c:\Users\user\OneDrive\Desktop\churn
streamlit run app.py
```

The app will open at **http://localhost:8501**

### App Features
- 🎛️ **Sidebar inputs** — all 19 customer features with sliders & dropdowns
- 🔮 **Real-time prediction** — churn / no-churn with probability %
- 📊 **Model metrics** — accuracy, precision, recall, F1
- ⚠️ **Risk factor chart** — top drivers of churn
- 💡 **Retention tips** — actionable business recommendations
- 🌑 **Dark glassmorphism UI** — modern gradient design

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA visualizations |
| `scikit-learn` | ML models, preprocessing, metrics |
| `joblib` | Model serialization |
| `streamlit` | Web app framework |

---

## 📊 Dataset Description

| Feature | Type | Description |
|---|---|---|
| `customerID` | string | Unique customer identifier |
| `gender` | categorical | Male / Female |
| `SeniorCitizen` | binary | 1 if senior citizen |
| `Partner` | categorical | Has partner Yes/No |
| `Dependents` | categorical | Has dependents Yes/No |
| `tenure` | numeric | Months with company |
| `PhoneService` | categorical | Has phone service |
| `MultipleLines` | categorical | Multiple phone lines |
| `InternetService` | categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | categorical | Has online security |
| `OnlineBackup` | categorical | Has online backup |
| `DeviceProtection` | categorical | Has device protection |
| `TechSupport` | categorical | Has tech support |
| `StreamingTV` | categorical | Streams TV |
| `StreamingMovies` | categorical | Streams movies |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | categorical | Paperless billing Yes/No |
| `PaymentMethod` | categorical | Payment type |
| `MonthlyCharges` | numeric | Monthly bill amount |
| `TotalCharges` | numeric | Total amount charged |
| `Churn` | target | Customer churned Yes/No |

---

## 🔑 Key Findings

1. **Contract type** is the strongest predictor — month-to-month customers churn 3× more
2. **Fiber optic** users have higher churn despite faster speeds (likely price sensitivity)
3. **New customers** (tenure < 6 months) are highest risk
4. **Electronic check** payment correlates strongly with churn
5. Customers **without online security or tech support** churn significantly more

---

*Built with Python · scikit-learn · Streamlit*
