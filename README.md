# 📊 Automated Detection of Data Drift and Model Performance Degradation

This repository contains a complete **Machine Learning Monitoring System** capable of:

✔️ Detecting distribution changes in input data (Data Drift)  
✔️ Detecting degradation in model performance after deployment  
✔️ Logging metrics into a database  
✔️ Visualizing results in an interactive dashboard  
✔️ Sending alerts (email/notifications) when issues occur  
✔️ Supporting multiple datasets (classification & regression)

This project was developed as part of a Bachelor's final project, and follows **industry-standard MLOps practices**.

---

## 🚀 Project Overview

Machine learning models often lose accuracy over time after deployment due to changing input data distributions or real-world conditions. This repository implements a modular monitoring pipeline that:

🔹 Loads reference and current production data  
🔹 Detects statistical drift using PSI (Population Stability Index)  
🔹 Evaluates model performance over time  
🔹 Stores metric history  
🔹 Provides alerts when thresholds are exceeded  
🔹 Shows results via an interactive dashboard

---

## 🧱 Repository Structure

```text
.
├── Train model/                       # Artifacts for Telco model
│   ├── model.pkl
│   ├── feature_selector.pkl
│   ├── all_features.csv
│   ├── selected_features.csv
│   ├── reference.csv
│   ├── current.csv
│   └── baseline_metrics.csv
├── Eth/                              # Artifacts for Ethereum model
│   ├── eth_1min.csv
│   ├── eth_1hour.csv
│   ├── train_model.py
│   ├── model.pkl
│   ├── feature_selector.pkl
│   ├── all_features.csv
│   ├── selected_features.csv
│   ├── reference.csv
│   ├── current.csv
│   └── baseline_metrics.csv
├── data_loader.py
├── dataset_adapters.py
├── drift_detection.py
├── main.py
├── model_monitor.py
├── metrics_store.py
├── alert_system.py
├── config_loader.py
├── dashboard.py
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 📌 Features

### 📈 Automated Drift Detection

- Uses **Population Stability Index (PSI)** to measure distribution changes
- Detects drift for both numerical and categorical features
- Supports multiple dataset types through adapters

### 💡 Performance Monitoring

- Computes performance metrics over production data:
  - Classification: *Accuracy, F1, Precision, Recall*
  - Regression: *MAE, RMSE, R²*
- Compares against baseline (validation) metrics
- Logs historic performance

### 🛠️ Modular and Extensible

- Adapter-based inputs — add new datasets easily
- Separate data loader for each dataset
- Config-driven architecture (`config.yaml`)
- Designed for multiple tasks (classification & regression)

### 📊 Dashboard

Built with **Streamlit** and **Plotly** to visualize:

- Drift trends over time
- Performance metric trends
- Alerts and status badges
- Raw metric logs

---

## 📄 Folder Descriptions

### `Train model/`

Contains Telco customer churn artifacts from training:

- `model.pkl`: Trained classification model  
- `feature_selector.pkl`: Feature selector  
- `reference.csv`: Reference (training) data  
- `current.csv`: Latest data  
- `all_features.csv`: One-hot features  
- `selected_features.csv`: Post-selection features  
- `baseline_metrics.csv`: Validation reference metrics 

---

### `Eth/`

Contains Ethereum price model artifacts:

- Raw datasets (`eth_1min.csv`, `eth_1hour.csv`)
- Trained regression model and selector
- Reference & current production splits
- Feature lists & baseline metrics

---

## 🧪 How to Train Models

### ☑️ Train Telco Model

```bash
python "Train model/train_model.py"
```

### ☑️ Train Ethereum Model (1-hour)

```bash
python Eth/train_model.py
```

This will generate all required artifacts for monitoring.

---

## ▶️ Running the Monitoring Pipeline

Configure your dataset in `config.yaml`:

```yaml
app:
  dataset_name: "ethereum"         # "telco" or "ethereum"
model:
  path: "Eth/model.pkl"
  baseline_metrics_path: "Eth/baseline_metrics.csv"
monitoring:
  drift_threshold: 0.2
  performance_drop_threshold: 0.05
alerts:
  email_enabled: false
```

Then run:

```bash
python main.py
```

This script performs:

* Drift Detection
* Performance Evaluation
* Metrics Logging
* Optional Alerts

---

## 📊 Dashboard Visualization

Start the dashboard with:

```bash
streamlit run dashboard.py
```

Access it at:

```
http://localhost:8501
```

The dashboard displays:

✔ Drift heatmaps
✔ Performance curves
✔ Current status badges
✔ Raw metric logs

---

## 📩 Alerting & Notifications

The system supports email alerts via SMTP (e.g., Gmail). Use an **App Password** for security.

Configure in `config.yaml`:

```yaml
alerts:
  email_enabled: true
  sender_email: "example@gmail.com"
  sender_password: "app_password_here"
  receiver_email: "notify@domain.com"
```

Alerts trigger when:

✔ Drift > threshold
✔ Performance drop > threshold

---

## 📝 Example Config (config.yaml)

```yaml
app:
  dataset_name: "telco"
model:
  path: "Train model/model.pkl"
  baseline_metrics_path: "Train model/baseline_metrics.csv"
monitoring:
  drift_threshold: 0.2
  performance_drop_threshold: 0.05
alerts:
  email_enabled: false
dashboard:
  host: "localhost"
  port: 8501
```

---

## 🧠 Why This Matters

This system demonstrates:

✔ Real-time model monitoring
✔ Adaptation for multiple model types
✔ Modular, extensible pipeline
✔ Data and model drift handling
✔ End-to-end visualization

It’s designed for **production readiness** and academic research use.

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Cite / References

If using this in academic work, reference:

✔ Population Stability Index (PSI) for drift
✔ Regression & Classification monitoring metrics

---

## 📬 Contact

✨ Created by **Tina Hafezi**
📍 Bachelor Final Project — ML Model Monitoring System
