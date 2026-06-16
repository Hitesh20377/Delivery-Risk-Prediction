# 🚚 Predictive Delivery Risk Intelligence System

A Machine Learning-based logistics analytics platform that predicts shipment delay risk before dispatch and helps operations teams prioritize high-risk orders using predictive intelligence.

---

## 📌 Project Overview

Late deliveries lead to customer dissatisfaction, SLA breaches, and increased operational costs.

This project uses Machine Learning to:

- Predict delivery delay risk before shipment
- Identify high-risk orders
- Analyze risky regions and shipping modes
- Quantify sales and profit exposure
- Provide actionable operational recommendations

---

## 🎯 Business Problem

APL Logistics requires a proactive system that can:

- Detect potential shipment delays early
- Prioritize risky shipments
- Improve customer communication
- Reduce operational disruption
- Support data-driven logistics planning

---

## 📊 Dataset

**Source:**

:contentReference[oaicite:0]{index=0}

Dataset contains:

- Order Information
- Shipping Details
- Customer Segments
- Markets & Regions
- Product Data
- Sales & Profit Metrics
- Delivery Status

---

## 🛠️ Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn
- Random Forest Classifier

### Visualization

- Plotly
- Matplotlib

### Dashboard

- Streamlit

---

## 🤖 Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature engineering
- Categorical encoding
- Feature selection

### Model

Random Forest Classifier

### Outputs

- Delay Probability
- Risk Category
- Operational Recommendation

---

## 📈 Dashboard Modules

### Executive Summary

- Total Orders
- Delay Rate
- Sales at Risk
- Profit at Risk
- Business Recommendations

### Delay Risk Analysis

- Delayed vs Non-Delayed Orders
- Customer Segment Risk
- Shipping Performance Analysis

### Operations Action Panel

- High-Risk Order Queue
- Risk Categorization
- Delay Probability Distribution
- Operational Action Recommendations

### Model Performance

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

---

## 📸 Dashboard Screenshots

### Executive Summary

![Executive Summary](screenshots/executive_summary.png)

### Risk Overview

![Risk Overview](screenshots/risk_overview.png)

### Operations Action Panel

![Operations Action Panel](screenshots/operations_action_panel.png)

---

## 📁 Project Structure

```text
Project 2/
│
├── dashboard/
│   ├── app.py
│   └── style.css
│
├── data/
│
├── models/
│   ├── delivery_risk_model.pkl
│   └── model_comparison.csv
│
├── reports/
│   └── figures/
│
├── screenshots/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predictor.py
│   ├── evaluator.py
│   ├── risk_scoring.py
│   ├── visualizer.py
│   └── kpi_calculator.py
│
├── main.py
├── requirement.txt
└── README.md
```

---

## 🚀 Key Business Insights

- More than half of shipments show delay risk.
- Certain markets and regions exhibit consistently higher delay probability.
- Risk scoring enables proactive shipment prioritization.
- Early intervention can reduce customer dissatisfaction and operational losses.

---

## 🔮 Future Improvements

- XGBoost Integration
- LightGBM Integration
- Hyperparameter Tuning
- Real-Time Prediction APIs
- Cloud Deployment
- Automated Alert System

---

## 👨‍💻 Author

**Hitesh Kandpal**

B.Tech CSE Student  
Machine Learning • Data Analytics • Predictive Modeling

GitHub: https://github.com/Hitesh20377
