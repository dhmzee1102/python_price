# PUTRA AgriTrack - Price Prediction API 🌾

This repository contains the backend service for the **AI-Powered UPM Agri-Monitoring Dashboard**. It provides a RESTful API to predict agricultural product prices using a trained Scikit-learn model and manages item/premise metadata.

## 📊 Dataset Information
The machine learning model is trained using datasets provided by **OpenDOSM** via the PriceCatcher initiative:
* **Premise Lookup:** [lookup_premise.parquet](https://storage.data.gov.my/pricecatcher/lookup_premise.parquet)
* **Item Lookup:** [lookup_item.parquet](https://storage.data.gov.my/pricecatcher/lookup_item.parquet)
* **Transactional Records:** [pricecatcher_2026-01.parquet](https://storage.data.gov.my/pricecatcher/pricecatcher_2026-01.parquet)

## 🚀 Features
* **AI Price Prediction**: Post-processed estimates based on item code, state, and district.
* **Flutter-Optimized Endpoints**: Data models are pre-mapped (e.g., `item` to `name`) to match Flutter frontend requirements.
* **Automated Data Loading**: Loads model artifacts and metadata (premises/items) on startup from `price_model.pkl`.
* **Flexible Inputs**: Handles both String and Integer IDs to prevent 422 validation errors during frontend integration.
* **CORS Enabled**: Configured for cross-origin requests, allowing seamless connection with web and mobile dashboards.

## 🛠 Tech Stack
* **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) & [Joblib](https://joblib.readthedocs.io/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/)
* **Server:** [Uvicorn](https://www.uvicorn.org/)

## 📂 Project Structure
* `price_api.py`: Main FastAPI application logic and endpoint definitions.
* `price_model.pkl`: Compressed model artifacts including the trained pipeline, item lists, and premise data.
* `requirements.txt`: Python dependencies required to run the service.

