from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#Nuevo modelo
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_DATA = DATA_DIR / "train.csv"
MODEL_FILE_LR = MODELS_DIR / "modelo_churn_lr.pkl" # Se ha renombrado el modelo modelo_churn_rf.pkl a modelo_churn_lr.pkl para reflejar el uso de LogisticRegression
MODEL_FILE_RF = MODELS_DIR / "modelo_churn_rf.pkl" # Nuevo modelo RandomForestClassifier


def entrenar_modelos():
    """
    Entrena dos modelos: LogisticRegression y RandomForestClassifier.
    """

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/train.csv. Primero ejecuta src/preparar_datos.py"
        )

    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # 1. Regresión Logística
    modelo_lr = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", LogisticRegression())
        ]
    )

    modelo_lr.fit(X, y)

    joblib.dump(modelo_lr, MODEL_FILE_LR)

    print("Modelo de Regresión Logística entrenado correctamente.")
    print(f"Modelo de Regresión Logística guardado en: {MODEL_FILE_LR}")

    # 2. Random Forest (NUEVO ALGORITMO)
    modelo_rf = Pipeline(
        steps=[
            ("escalado", StandardScaler()),
            ("clasificador", RandomForestClassifier(random_state=42))
        ]
    )
    modelo_rf.fit(X, y)
    joblib.dump(modelo_rf, MODEL_FILE_RF)
    
    print("Modelo de Random Forest entrenado correctamente.")
    print(f"Modelo de Random Forest guardado en: {MODEL_FILE_RF}")

if __name__ == "__main__":
    entrenar_modelos()
