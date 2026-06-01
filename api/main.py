from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE_LR = BASE_DIR / "models" / "modelo_churn_lr.pkl"
MODEL_FILE_RF = BASE_DIR / "models" / "modelo_churn_rf.pkl"

app = FastAPI(
    title="API de Predicción de Churn",
    version="0.2.0",
    description="API básica para consumir dos modelos de Machine Learning."
)

class Cliente(BaseModel):
    edad: int
    antiguedad_meses: int
    saldo_promedio: float
    reclamos: int
    usa_app: int

def cargar_modelo(modelo_path):
    """
    Carga el modelo entrenado si existe.
    """
    if not modelo_path.exists():
        return None

    return joblib.load(modelo_path)

@app.get("/")
def inicio():
    return {
        "mensaje": "API de predicción de churn activa",
        "modelos_disponibles":["logistic_regression", "random_forest"]
    }

@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelos": {
            "logistic_regression": MODEL_FILE_LR.exists(),
            "random_forest": MODEL_FILE_RF.exists()
        }
    }

@app.post("/predict/{modelo}")
def predict(modelo: str, cliente: Cliente):
    """
    Predice churn usando el modelo especificado:
    - 'logistic_regression' para Regresión Logística
    - 'random_forest' para Random Forest
    """
    
    if modelo == "logistic_regression":
        model_path = MODEL_FILE_LR
        model_name = "Regresión Logística"
    elif modelo == "random_forest":
        model_path = MODEL_FILE_RF
        model_name = "Random Forest"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{modelo}' no válido. Usar 'logistic_regression' o 'random_forest'"
        )

    modelo_ml = cargar_modelo(model_path)

    if modelo_ml is None:
        raise HTTPException(
            status_code=503,
            detail=f"El modelo {model_name} aún no está disponible. Ejecuta src/entrenar_modelo.py"
        )

    datos = pd.DataFrame([cliente.model_dump()])

    prediccion = int(modelo_ml.predict(datos)[0])

    probabilidad = None
    if hasattr(modelo_ml, "predict_proba"):
        probabilidad = float(modelo_ml.predict_proba(datos)[0][1])

    return {
        "modelo_utilizado": model_name,
        "churn_predicho": prediccion,
        "probabilidad_churn": probabilidad
    }
