# Titanic Survival Prediction Deployment

A machine learning deployment project that predicts survival chances on the Titanic using FastAPI and Streamlit.

## Project Structure
```
PMLDL-Assignment-1
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api (FastAPI server)
│   │   ├── app (Streamlit application) 
│   │   └── docker-compose.yml
│   └── models
│   │   └── train_model.py
└── models (trained model files)
```


## How to Run

```bash
python -m pip install scikit-learn pandas numpy seaborn
python code/models/train_model.py
```

```bash
cd code/deployment
docker-compose up --build
```

Streamlit app: http://localhost:8501

API documentation: http://localhost:8000/docs
