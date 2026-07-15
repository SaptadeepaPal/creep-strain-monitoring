#  Creep Life Prediction & Intelligent Early Warning System for P91 Steel using Machine Learning
### AI-Driven Non-Destructive Monitoring of High Temperature Power Plant Components

<p align="center">

**Internship Project @ CSIR – National Metallurgical Laboratory (CSIR-NML), Jamshedpur**

*Applied Machine Learning | Non-Destructive Testing (NDT) | Materials Engineering | Predictive Maintenance*

</p>

---

## Project Overview

Power plants operate under extremely harsh conditions where steam flows continuously through boiler tubes and pipelines at **very high temperatures (500–650°C)** and **high pressure**. Over thousands of operating hours, these components undergo **creep deformation**, a slow and permanent elongation of the material due to prolonged thermal stress.
If creep damage is not detected early, it may eventually lead to **tube rupture**, resulting in:

- Unexpected plant shutdowns
- Huge maintenance costs
- Safety hazards
- Loss of power generation

Traditionally, creep damage is evaluated using **Destructive Testing (DT)**, where a section of the tube is physically cut out for laboratory examination. Although accurate, this method permanently damages the component, is expensive, and cannot provide continuous monitoring.

This project focuses on **Non-Destructive Testing (NDT)** using **Electromagnetic Eddy Current sensing technology**, allowing creep deformation to be monitored continuously **without damaging the material**.
Using real experimental data collected during my internship at **CSIR-NML's Creep Laboratory**, I developed a complete Machine Learning pipeline that predicts creep strain and intelligently detects when the material is approaching failure.

---

## Prediction Dashboard
<img width="1852" height="897" alt="image" src="https://github.com/user-attachments/assets/b0957c60-079a-4e95-a237-cea9e82e82f2" />

<img width="1832" height="910" alt="image" src="https://github.com/user-attachments/assets/ca41b715-8807-4776-8dd1-6782c92097a8" />


---

## Live Demo
https://creep-strain-monitoring-rkkncth4wxrxu9v6rvcfdw.streamlit.app/

# Industrial Background

## What is Creep?
Creep is the slow, time-dependent plastic deformation of a material subjected to constant stress at elevated temperatures.

A typical creep curve consists of three distinct stages:

### Primary Creep
- High initial strain rate
- Material undergoes work hardening
- Deformation rate gradually decreases

### Secondary Creep
- Nearly constant strain rate
- Stable deformation
- Longest operating region

### Tertiary Creep
- Rapid acceleration of strain
- Internal micro-crack formation
- Material approaches rupture

The transition from secondary to tertiary creep is the most critical phase because it provides the last opportunity to replace the component before catastrophic failure.

# Screenshots

## Creep Curve
<img width="751" height="452" alt="image" src="https://github.com/user-attachments/assets/820f60cb-67d0-43db-b211-695825197c13" />

---


# Eddy Current Based Sensor

The experimental setup uses an **electromagnetic sensor** positioned on the surface of the P91 steel specimen during creep testing.

## Working Principle

An alternating current passing through the probe coil produces a changing magnetic field.

This magnetic field induces **eddy currents** inside the conductive metal.

As creep deformation progresses:

- Microstructural changes occur
- Grain boundaries evolve
- Dislocation density changes
- Electrical conductivity changes
- Magnetic permeability changes

These changes affect the eddy current distribution, which in turn alters the electrical characteristics of the sensor.
The sensing circuit records the resulting **output voltage**, which correlates with the creep deformation occurring inside the material.
Instead of physically cutting the specimen, deformation can therefore be estimated directly from the sensor response.

---

# Dataset

The dataset was collected during controlled creep experiments conducted in the **CSIR-NML Creep Laboratory**.

Each record contains:

| Feature | Description |
|----------|-------------|
| Exposure Duration (Hours) | Time under constant load and temperature |
| Output Voltage | Voltage measured by the Eddy Current sensor |
| % Strain | Actual creep strain measured during the experiment |

The complete dataset contains the entire creep lifecycle including:

- Primary creep
- Secondary creep
- Tertiary creep
---

# Major Challenge in the Dataset

One interesting challenge observed during experimentation was the **highly non-uniform distribution of samples across different creep stages.**

The dataset contains:
- A very large number of observations in the Primary region
- Moderate observations in the Secondary region
- Very limited observations in the Tertiary region

This imbalance occurs naturally because:

- Primary and Secondary creep last for a very long time.
- Tertiary creep progresses very rapidly just before rupture.

As a result, machine learning models have relatively fewer examples from the most critical failure region, making prediction near rupture significantly more challenging.

---

# Machine Learning Pipeline

## 1. Data Preprocessing

Performed detailed exploratory analysis to understand:

- Missing values
- Duplicate records
- Outliers inspection
- Feature distributions
- Range of sensor voltage
- Strain evolution over time
- Relationship between voltage and creep deformation

---

The raw experimental data was cleaned by:

- Removing duplicate observations
- Checking missing values
- Correcting inconsistent data types
- Verifying numerical ranges
- Sorting observations according to exposure duration

---

## 3. Feature Engineering

Important derived information included:

- Time progression ordering
- Trend analysis of strain evolution
- Consistent numerical formatting
- Preparation of model-ready features

Input Features:

- Exposure Duration
- Output Voltage

Target:

- Percentage Strain

---


## 5. Dataset Splitting

The dataset was divided into:

- Training Set
- Validation Set
- Testing Set

to ensure unbiased evaluation and proper generalization.

---

# Models

Three regression models were developed and evaluated.

## Model 1

Random Forest Regressor

Strengths:

- Robust against noise
- Good baseline model
- Handles non-linear relationships

---

## Model 2

Gradient Boosting Regressor

Strengths:

- Sequential error correction
- Better fitting of complex curves
- Improved performance over Random Forest

---

## Model 3 (Final Selected Model)

# XGBoost Regressor

XGBoost achieved the best balance between:

- Prediction accuracy
- Generalization
- Stability
- Low prediction error

and was therefore selected as the final production model.

---

# Model Performance

| Model | RMSE ↓ | MAE ↓ | R² Score ↑ |
|--------|---------|---------|-----------|
| Random Forest | 0.241 | 0.182 | 0.995 |
| Gradient Boosting | 0.167 | 0.121 | 0.998 |
| **XGBoost** | **0.1089** | **0.079** | **0.999** |

The final XGBoost model successfully captured the highly nonlinear behaviour of creep deformation throughout all three creep stages while maintaining the lowest prediction error.

---

# Intelligent Alert System

Predicting strain alone is often insufficient in industrial environments.

Operators require immediate interpretation of the predicted condition.

To address this, an intelligent rule-based alert engine was developed.

The system continuously analyzes the predicted creep trend.

### Normal

Material is operating in stable Primary or Secondary creep.

Status:

✅ Normal

---

### Warning

When the model detects an accelerated increase in strain indicating departure from stable creep behaviour,

Status:

⚠️ Warning

This serves as an early maintenance indicator.

---

### Critical

When the predicted strain enters the rapidly accelerating tertiary creep region,

Status:

🚨 Critical

This indicates that the material is approaching rupture and immediate inspection or replacement should be considered.

---

# Deployment

A complete interactive web application was developed where the user provides:

- Exposure Duration
- Sensor Output Voltage

The system instantly predicts:

- Estimated % Strain
- Warning Status

making the model suitable for real-time industrial monitoring demonstrations.

---

# Project Architecture

```
Eddy Current Sensor
        │
        ▼
Voltage + Exposure Time
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Model Training
(Random Forest,
Gradient Boosting,
XGBoost)
        │
        ▼
Model Evaluation
        │
        ▼
Best Model (XGBoost)
        │
        ▼
Strain Prediction
        │
        ▼
Trend Analysis
        │
        ▼
Alert Generation
(Normal / Warning / Critical)
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Streamlit

---

# Results

✅ Accurate prediction of creep strain using Eddy Current sensor measurements

✅ Successful modelling of nonlinear creep behaviour

✅ Automated early warning generation before material failure

✅ Low prediction error (RMSE: **0.1089 % strain**)

✅ End-to-end deployment through an interactive Streamlit application

---

# Future Improvements

- Incorporate additional sensor modalities such as temperature and load for multi-sensor fusion.
- Explore sequence-based deep learning models (LSTM/Transformer) to better capture temporal creep evolution.
- Quantify prediction uncertainty, particularly in the tertiary creep region where data availability is limited.
- Validate the framework on different high-temperature alloys beyond P91 steel.
- Integrate real-time sensor streaming for continuous online monitoring in industrial environments.

---

# Acknowledgement

This project was completed during my internship at **CSIR – National Metallurgical Laboratory (CSIR-NML), Jamshedpur**, where I had the opportunity to work with experimental creep testing data collected using an Eddy Current based Non-Destructive Evaluation system. The project combines materials engineering, non-destructive testing, and machine learning to demonstrate how AI can support predictive maintenance and improve the reliability of critical power plant components.
