"""
API de predicción de churn con FastAPI.

La API carga dos modelos serializados, valida los datos de entrada
y devuelve dos predicciones junto con sus probabilidades.
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH_LR = PROJECT_ROOT / "models" / "modelo_churn_lr_v1.joblib"
MODEL_PATH_RF = PROJECT_ROOT / "models" / "modelo_churn_rf_v1.joblib"

VERSION_MODELO_LR = "modelo_churn_lr_v1"
VERSION_MODELO_RF = "modelo_churn_rf_v1"
AUTOR = "Noel Andia Flores"

if not MODEL_PATH_LR.exists():
    raise RuntimeError(
        "No se encontró el modelo de Regresión Logística serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

if not MODEL_PATH_RF.exists():
    raise RuntimeError(
        "No se encontró el modelo de Random Forest serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo_lr = joblib.load(MODEL_PATH_LR)
modelo_rf = joblib.load(MODEL_PATH_RF)

class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses numéricos",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )

class PrediccionModelo(BaseModel):
    modelo: str
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str

class PrediccionSalida(BaseModel):
    modelos_evaluados: str
    riesgo: str
    prediccion: str
    probabilidad: str
    descripcion: str
    recomendacion: str
    version_modelo: str
    autor: str


app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version="2.0.0",
)

@app.get("/")
def inicio() -> dict[str, object]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "modelos_disponibles": ["logistic_regression", "random_forest"],
        "autor": AUTOR,
    }

@app.get("/health")
def health() -> dict[str, object]:
    return {
        "estado": "ok",
        "modelos": {
            "logistic_regression": VERSION_MODELO_LR,
            "random_forest": VERSION_MODELO_RF,
        },
    }

@app.get("/info")
def info() -> dict[str, object]:
    return {
        "Version": VERSION_MODELO_LR + " / " + VERSION_MODELO_RF,
        "autor": AUTOR,
        "Variables_entrada": {
            "antiguedad": "Meses como cliente (int)",
            "cargo_mensual": "Monto mensual pagado (float)",
            "reclamos": "Cantidad de reclamos recientes (int)",
        },
    }


def _evaluar_modelo(modelo, nombre_modelo: str, version_modelo: str, X: list[list[float]]) -> PrediccionModelo:
    probabilidad = float(modelo.predict_proba(X)[0][1])
    if probabilidad >= 0.75:
        etiqueta = "muy_alto_riesgo"
    elif probabilidad >= 0.50 and probabilidad < 0.75:
        etiqueta = "alto_riesgo"
    elif probabilidad >= 0.25 and probabilidad < 0.50:
        etiqueta = "riesgo_moderado"
    else:
        etiqueta = "bajo_riesgo"

    return PrediccionModelo(
        modelo=nombre_modelo,
        prediccion=etiqueta,
        probabilidad=round(probabilidad, 4),
        version_modelo=version_modelo,
        autor=AUTOR,
    )

@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        prediccion_lr = _evaluar_modelo(
            modelo=modelo_lr,
            nombre_modelo="logistic_regression",
            version_modelo=VERSION_MODELO_LR,
            X=X,
        )

        prediccion_rf = _evaluar_modelo(
            modelo=modelo_rf,
            nombre_modelo="random_forest",
            version_modelo=VERSION_MODELO_RF,
            X=X,
        )

        modelos = prediccion_lr.modelo + " / " + prediccion_rf.modelo
        if prediccion_lr.prediccion == prediccion_rf.prediccion:
            riesgo = prediccion_lr.prediccion
        elif prediccion_rf.prediccion == "muy_alto_riesgo" and prediccion_lr.prediccion == "alto_riesgo":
            riesgo = prediccion_rf.prediccion
        elif prediccion_rf.prediccion == "alto_riesgo" and prediccion_lr.prediccion == "muy_alto_riesgo":
            riesgo = prediccion_lr.prediccion
        elif prediccion_rf.prediccion == "alto_riesgo" and prediccion_lr.prediccion == "riesgo_moderado":
            riesgo = prediccion_rf.prediccion
        elif prediccion_rf.prediccion == "riesgo_moderado" and prediccion_lr.prediccion == "alto_riesgo":
            riesgo = prediccion_lr.prediccion
        else:
            riesgo = "riesgo_moderado"

        if prediccion_lr.probabilidad == prediccion_rf.probabilidad:
            probabilidad = f"{round(prediccion_lr.probabilidad*100, 2)} %"
        elif prediccion_rf.probabilidad > prediccion_lr.probabilidad:
            probabilidad = f"{round(prediccion_lr.probabilidad*100, 2)} - {round(prediccion_rf.probabilidad*100, 2)} %"
        else:
            probabilidad = f"{round(prediccion_rf.probabilidad*100, 2)} - {round(prediccion_lr.probabilidad*100, 2)} % "

        if riesgo == "muy_alto_riesgo":
            recomendacion = ("Se recomienda contactar al cliente de inmediato para ofrecerle beneficios exclusivos o"
                             " resolver posibles problemas que puedan estar enfrentando de manera prioritaria.")
        elif riesgo == "alto_riesgo":
            recomendacion = ("Se recomienda contactar al cliente para ofrecerle beneficios o"
                             " resolver posibles problemas que puedan estar enfrentando.")
        elif riesgo == "riesgo_moderado":
            recomendacion = ("Se recomienda monitorear al cliente y ofrecerle beneficios o"
                             " resolver posibles problemas que puedan estar enfrentando.")
        else:
            recomendacion = ("Se recomienda mantener una buena relación con el cliente y"
                             " ofrecerle beneficios para fidelizarlo.")

        return PrediccionSalida(
            modelos_evaluados=modelos,
            riesgo=riesgo,
            prediccion=f"{prediccion_lr.prediccion} / {prediccion_rf.prediccion}",
            probabilidad=probabilidad,
            descripcion=(
                "Se han utilizado los modelos de regresión logística y bosque aleatorio, "
                "para evaluar el riesgo de churn del cliente."
            ),
            recomendacion=recomendacion,
            version_modelo=f"{prediccion_lr.version_modelo} / {prediccion_rf.version_modelo}",
            autor=AUTOR,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar las predicciones de churn.",
        ) from exc