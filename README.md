# 🚚 Predictive Delivery Risk Intelligence System

An end-to-end Machine Learning solution designed to predict shipment delay risk in logistics operations before deliveries occur.

This project helps logistics and supply chain teams identify high-risk shipments, prioritize operational actions, reduce delivery failures, and improve customer satisfaction through predictive analytics and interactive business intelligence dashboards.

---

## 📌 Business Problem

In modern supply chain operations, late deliveries can result in:

- SLA breaches
- Customer dissatisfaction
- Revenue loss
- Increased operational costs
- Reactive last-minute interventions

Most organizations detect delivery issues only after delays occur.

This project provides a proactive solution by predicting delivery risk before shipment completion, enabling logistics teams to take preventive actions.

---

## 🎯 Project Objectives

- Predict late delivery risk using Machine Learning
- Generate shipment-level risk scores
- Identify high-risk regions and markets
- Analyze shipping mode performance
- Monitor sales and profit exposure
- Support operational decision-making through dashboards

---

## 🏢 Industry Use Case

### Client Scenario
APL Logistics (KWE Group)

### Business Need
The organization requires:

- Early identification of risky shipments
- Operational prioritization of high-risk orders
- Visibility into delay drivers
- Risk-based decision making

### Solution Delivered

A Predictive Delivery Risk Intelligence System featuring:

- Machine Learning prediction engine
- Interactive Streamlit dashboard
- Operational action panel
- Executive reporting and analytics

---

## 🛠 Technology Stack

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- Random Forest Classifier

### Visualization
- Plotly
- Matplotlib

### Deployment
- Streamlit

### Model Serialization
- Joblib

---

## 📂 Project Structure

```text
Delivery-Risk-Prediction/
│
├── dashboard/
│   ├── app.py
│   └── style.css
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_config.py
│   ├── model_training.py
│   ├── predictor.py
│   ├── evaluator.py
│   ├── visualizer.py
│   ├── risk_scoring.py
│   └── kpi_calculator.py
│
├── models/
│   ├── delivery_risk_model.pkl
│   └── model_comparison.csv
│
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebook/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Modules

### Executive Overview

Provides a high-level summary of:

- Total Orders
- Delayed Orders
- Delay Rate
- Sales at Risk
- Profit at Risk
- Average Shipping Time

---

### Executive Summary

Business-focused insights including:

- Risk overview
- Most risky regions
- Most risky shipping modes
- Strategic recommendations

---

### Delay Risk Analysis

Analyzes:

- Real shipping days
- Scheduled shipping days
- Delay gap patterns
- Sales impact of delays
- Profit impact of delays

---

### Region & Market Analysis

Provides:

- Risk by market
- Risk by country
- Market performance
- Risk heatmaps

---

### Shipping Mode Analysis

Evaluates:

- Delay risk by shipping mode
- Shipping performance
- Sales contribution
- Profit contribution

---

### Order-Level Prediction

Allows users to:

- Predict delay risk for new shipments
- Evaluate existing orders
- Generate risk scores
- Receive operational recommendations

---

### Operations Action Panel

Provides:

- High-risk shipment queue
- Risk categorization
- Sales exposure analysis
- Operational recommendations

---

## ⚙️ Machine Learning Pipeline

### Data Preparation

- Missing value handling
- Feature engineering
- Data cleaning
- Risk feature generation

### Feature Processing

- Numerical feature handling
- One-Hot Encoding
- ColumnTransformer pipeline

### Model

Random Forest Classifier

Configuration:

- n_estimators = 30
- max_depth = 12
- class_weight = balanced
- random_state = 42

---

## 📈 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 67.94% |
| Precision | 73.33% |
| Recall | 65.27% |
| F1 Score | 69.07% |

---

## 📉 Visual Outputs

Generated reports include:

- Confusion Matrix
- ROC Curve
- Feature Importance Analysis
- Model Comparison Report

---

## 🚨 Business Impact

The solution helps logistics managers:

### Risk Management

- Identify high-risk shipments before delivery
- Reduce shipment failures

### Revenue Protection

- Monitor sales exposure
- Reduce profit leakage

### Operational Efficiency

- Prioritize shipments
- Allocate resources effectively

### Customer Experience

- Improve communication
- Reduce customer dissatisfaction

---

## 📚 Dataset

Dataset Used:

### DataCo Smart Supply Chain for Big Data Analysis

Kaggle Dataset:

https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Due to GitHub file size limitations, raw and processed datasets are not included in this repository.

Place the downloaded dataset inside:

```text
data/raw/
```

Example:

```text
data/raw/DataCoSupplyChainDataset.csv
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/Hitesh20377/Delivery-Risk-Prediction.git
```

### Navigate to Project

```bash
cd Delivery-Risk-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run:

```bash
python main.py
```

This will:

- Clean the dataset
- Generate engineered features
- Train the model
- Save the trained model
- Generate evaluation reports

---

## 📊 Launch Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

Dashboard will be available at:

```text
http://localhost:8501
```

---

## 📷 Dashboard Screenshots

Add screenshots here after uploading:

### Executive Overview

![Executive Overview](reports/figures/dashboard_overview.png)

### Risk Analysis

![Risk Analysis](reports/figures/risk_analysis.png)

### Operations Action Panel

![Operations Panel](reports/figures/operations_panel.png)

---

## 🔮 Future Improvements

Potential enhancements include:

- XGBoost implementation
- LightGBM implementation
- Hyperparameter tuning
- Cross-validation framework
- Explainable AI (SHAP)
- Live API integration
- Real-time shipment monitoring
- Cloud deployment

---

## 👨‍💻 Author

### Hitesh Kandpal

B.Tech Computer Science Engineering

Skills:

- Machine Learning
- Data Analytics
- Python
- SQL
- Streamlit
- Business Intelligence

GitHub:

https://github.com/Hitesh20377

LinkedIn:

(Add your LinkedIn Profile URL)

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.

It helps others discover the project and supports future development.
