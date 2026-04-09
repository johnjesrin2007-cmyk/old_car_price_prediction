# 🚀 MLOps Pipeline for Used Car Price Prediction

## 📌 Project Overview

This project is an end-to-end **MLOps pipeline** that predicts the price of used cars based on various features such as brand, model, mileage, engine capacity, and more.

It simulates a real-world system similar to platforms like OLX and Cars24, where machine learning models are deployed to estimate product prices dynamically.

---

## 🎯 Objectives

* Build a robust **machine learning model** for price prediction
* Implement a **data preprocessing pipeline**
* Track experiments using **MLflow**
* Serve predictions via a **FastAPI API**
* Automate workflows using **CI/CD (GitHub Actions)**
* Containerize the application using **Docker**

---

## 🧠 Problem Statement

Predict the **price of a used car** given structured input data:

* Car specifications (engine, mileage, power)
* Ownership details
* Location and seller type

---

## 📂 Project Structure

```
.
├── data/
│   └── olx_cars_realistic.csv
├── model/
│   └── model.pkl
├── pipeline/
│   └── training_pipeline.py
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .github/workflows/
│   └── ci.yml
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* MLflow
* FastAPI
* Docker
* GitHub Actions

---

## 🔄 Workflow

### 1. Data Preprocessing

* Handle missing values using:

  * Mean imputation (numerical)
  * Mode imputation (categorical)
* Encode categorical variables using OneHotEncoder
* Scale numerical features using StandardScaler

---

### 2. Model Training

* Model: RandomForestRegressor
* Hyperparameter tuning using RandomizedSearchCV
* Cross-validation for evaluation
* Metrics:

  * MAE
  * MSE
  * RMSE
  * R² Score

---

### 3. Experiment Tracking

* All experiments are logged using MLflow:

  * Parameters
  * Metrics
  * Trained model artifacts

---

### 4. API Deployment

* Built using FastAPI
* Endpoint: `/predict`
* Accepts JSON input and returns predicted price

---

### 5. CI/CD Pipeline

Using GitHub Actions:

* Automatically runs training pipeline on push
* Verifies model creation
* Builds Docker image
* Pushes image to Docker Hub

---

## 🐳 Docker Usage

### Build Image

```
docker build -t mlops-pipeline .
```

### Run Container

```
docker run -p 8000:8000 mlops-pipeline
```

---

## 🌐 API Usage

### Start Server

```
uvicorn main:app --reload
```

### Test Endpoint

Open:

```
http://127.0.0.1:8000/docs
```

### Sample Input

```json
{
  "brand": "Hyundai",
  "model": "i20",
  "year": 2018,
  "km_driven": 40000,
  "fuel": "Petrol",
  "transmission": "Manual",
  "owner": 1,
  "location": "Chennai",
  "mileage": 18.5,
  "engine": 1197,
  "max_power": 82,
  "seats": 5,
  "seller_type": "Dealer"
}
```

---

## 📊 Example Output

```json
{
  "predicted_price": 562000
}
```

---

## 🧪 Running the Training Pipeline

```
python pipeline/training_pipeline.py
```

---

## 🚀 Key Features

* End-to-end ML pipeline
* Modular code structure
* Automated training and deployment
* Real-world dataset simulation
* Production-ready API

---

## 💡 Future Improvements

* Add model monitoring and logging
* Deploy on cloud platforms (AWS, Render, GCP)
* Add frontend interface
* Improve model performance with advanced tuning

---

## 🧑‍💻 Author

Developed as part of an MLOps learning project.

---

## ⭐ Conclusion

This project demonstrates a complete workflow from data preprocessing to deployment, following industry-level MLOps practices. It is designed to showcase practical skills required for machine learning internships and real-world applications.
