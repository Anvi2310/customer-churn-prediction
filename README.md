# Customer Churn Prediction

**Live demo:** https://customer-churn-prediction-49wb.onrender.com/

> Hosted on Render's free plan: if the site has been idle, the first load can take up to a minute while it wakes up.

Predicts whether a telecom customer is likely to churn (cancel their service), using a Random Forest classifier trained on the IBM Telco Customer Churn dataset. Predictions are served through a Flask web app: enter a customer's details in the form and it shows whether they are likely to churn or stay.

## Dataset

`Telco_dataset.csv` is the IBM Telco Customer Churn dataset:

- 7,043 customers, 19 input features + `Churn` target
- 5,174 customers stayed (73.5%), 1,869 churned (26.5%)
- Features cover demographics (gender, senior citizen, partner, dependents), account details (tenure, contract, billing, payment method, charges) and subscribed services (phone, internet, security, backup, streaming, etc.)

## Approach

1. **Cleaning:** dropped `customerID`; filled 11 blank `TotalCharges` values (customers with 0 tenure) with 0.
2. **Encoding:** label-encoded the 15 categorical columns; encoders saved to `encoders.pkl`.
3. **Split:** 80/20 train/test split (`random_state=42`).
4. **Class balancing:** SMOTE applied to the training set only.
5. **Model selection:** compared Decision Tree, Random Forest and XGBoost with 5-fold cross-validation. Random Forest scored highest.
6. **Final model:** Random Forest (100 trees) saved to `customer_churn_model.pkl`.

## Results

Evaluated on the 1,409-customer held-out test set:

| Metric | Value |
|---|---|
| Accuracy | **77.7%** |
| Churn precision | 0.58 |
| Churn recall | 0.58 |
| No-churn precision / recall | 0.85 / 0.85 |

Confusion matrix:

| | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actual: stay** | 880 | 156 |
| **Actual: churn** | 158 | 215 |

## Project structure

```
├── app.py                                    # Flask web app
├── templates/index.html                      # Prediction form UI
├── customer_churn_model.pkl                  # Trained Random Forest + feature names
├── encoders.pkl                              # Label encoders for categorical features
├── Customer_Churn_Prediction_using_ML.ipynb  # EDA, training and evaluation
├── Telco_dataset.csv                         # Dataset
├── requirements.txt                          # Python dependencies for the web app
└── .python-version                           # Python 3.12
```

## Run locally

Requires Python 3.12.

```bash
git clone https://github.com/Anvi2310/customer-churn-prediction.git
cd customer-churn-prediction
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:5001   # or: flask --app app run --port 5001
```

Open http://127.0.0.1:5001 in your browser.

> On macOS, port 5000 is often used by AirPlay Receiver, so the commands above use port 5001.

`scikit-learn` is pinned to 1.3.2 because the model was saved with that version. To re-run the notebook you will also need `matplotlib`, `seaborn`, `imbalanced-learn` and `xgboost`.

## Deployment

The app is deployed on [Render](https://render.com) as a Python web service at https://customer-churn-prediction-49wb.onrender.com/

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, Flask, Gunicorn
