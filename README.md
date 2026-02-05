
# 📊 Automated Detection of Data Drift and Model Performance Degradation

This repository contains a complete **Machine Learning Monitoring System** capable of:

✔️ Detecting distribution changes in input data (**Advanced Data Drift**)
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
🔹 Detects **multi-layer data drift** using statistical tests and model-behavior monitoring
🔹 Evaluates model performance over time
🔹 Stores metric history
🔹 Provides alerts when thresholds are exceeded
🔹 Shows results via an interactive dashboard

---

## 🧱 Repository Structure

```text
.
├── Train model/                       # Artifacts for Telco model
├── Eth/                              # Artifacts for Ethereum model
│
├── drift_detection/                  # 🔥 Advanced Drift Monitoring Engine
│   ├── drift_engine.py               # Main orchestrator for drift analysis
│   ├── statistical_drift.py          # KS Test, Wasserstein Distance, PSI
│   ├── data_quality.py               # Missing values, zero inflation, category drift
│   ├── embedding_drift.py            # Model output / probability drift
│   └── prediction_drift.py           # Prediction distribution & confidence shift
│
├── data_loader.py
├── dataset_adapters.py
├── main.py
├── model_monitor.py
├── metrics_store.py
├── risk_engine.py                    # System risk score computation
├── alert_system.py
├── config_loader.py
├── dashboard.py
├── fake_drift.py                     # Synthetic drift generator
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 📌 Features

### 📈 Advanced Automated Drift Detection

The system uses a **multi-layer drift detection framework** instead of a single metric:

| Drift Layer        | Method                           | Purpose                                  |
| ------------------ | -------------------------------- | ---------------------------------------- |
| Statistical Drift  | KS Test                          | Detects distribution shape change        |
| Statistical Drift  | Wasserstein Distance             | Measures magnitude of distribution shift |
| Statistical Drift  | PSI (Population Stability Index) | Monitors population stability            |
| Data Quality Drift | Missing/Zero/Category checks     | Detects data integrity issues            |
| Prediction Drift   | Output distribution monitoring   | Detects model behavior change            |
| Embedding Drift    | Probability space monitoring     | Detects concept drift signals            |

This provides **research-grade drift monitoring**, similar to production ML systems.

---

### 🧠 Dynamic Threshold Engineering

Instead of fixed thresholds only, the system supports:

```
Dynamic Threshold = Historical Mean + k·σ
```

Used to detect:

* Sudden anomalies
* Gradual degradation
* Concept drift trends

---

### 💡 Performance Monitoring

- Computes performance metrics over production data:
  - Classification: *Accuracy, F1, Precision, Recall*
  - Regression: *MAE, RMSE, R²*
- Compares against baseline (validation) metrics
- Logs historic performance

---

### 🛠️ Modular and Extensible


* Adapter-based inputs — add new datasets easily
* Separate data loader for each dataset
* Config-driven architecture (`config.yaml`)
* Designed for multiple tasks (classification & regression)
* **Pluggable drift detectors inside `drift_detection/`**

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
### 🧪 Synthetic Drift Simulation

The system includes a **drift injection module** to simulate real-world failures:

| Dataset  | Simulated Drift                                         |
| -------- | ------------------------------------------------------- |
| Telco    | Feature distribution shift, label noise, missing values |
| Ethereum | Price shocks, volatility spikes, zero inflation         |

Run:

```bash
python fake_drift.py
```

---

## ▶️ Running the Monitoring Pipeline

This script now performs:

* **Statistical Drift Detection (KS, Wasserstein, PSI)**
* **Data Quality Drift Detection**
* **Prediction / Concept Drift Monitoring**
* Performance Evaluation
* Risk Score Calculation
* Metrics Logging
* Optional Alerts

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

✔ Multi-layer ML drift monitoring
✔ Statistical + model-behavior monitoring
✔ Dynamic threshold engineering
✔ Data quality + prediction drift integration
✔ End-to-end monitoring architecture

This mirrors **real-world production ML monitoring platforms** and modern research approaches.

---

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
