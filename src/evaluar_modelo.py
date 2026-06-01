from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score # ROC AUC agregado para evaluación adicional

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

TEST_DATA = DATA_DIR / "test.csv"
MODEL_FILE_LR = MODELS_DIR / "modelo_churn_lr.pkl"
MODEL_FILE_RF = MODELS_DIR / "modelo_churn_rf.pkl"
METRICS_FILE = DOCS_DIR / "metricas_modelo.md"

def evaluar_modelos():
    """
    Evalúa ambos modelos entrenados y guarda las métricas principales.
    """

    if not TEST_DATA.exists():
        raise FileNotFoundError(
            "No se encontró data/test.csv. Primero ejecuta src/preparar_datos.py"
        )

    if not MODEL_FILE_LR.exists():
        raise FileNotFoundError(
            "No se encontró el modelo de Regresión Logística entrenado. Primero ejecuta src/entrenar_modelo.py"
        )
    
    if not MODEL_FILE_RF.exists():
        raise FileNotFoundError(
            "No se encontró el modelo de Random Forest entrenado. Primero ejecuta src/entrenar_modelo.py"
        )

    DOCS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(TEST_DATA)

    X_test = df.drop(columns=["churn"])
    y_test = df["churn"]

    modelos = {
        "Regresión Logística": MODEL_FILE_LR,
        "Random Forest": MODEL_FILE_RF
    }

    resultados = {}

    for nombre_modelo, path_modelo in modelos.items():
        if not path_modelo.exists():
            print(f"No se encontró el modelo {nombre_modelo}. Saltando evaluación.")
            continue

        modelo = joblib.load(path_modelo)

        y_pred = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # NUEVA MÉTRICA: ROC-AUC
        if hasattr(modelo, "predict_proba"):
            y_proba = modelo.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None

        resultados[nombre_modelo] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

        print(f"\n {nombre_modelo}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

    contenido = "# Métricas de modelos de churn\n\n"
    contenido += "## Comparativa de modelos\n\n"
    contenido += "| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |\n"
    contenido += "|--------|----------|-----------|--------|----------|---------|\n"
    
    for nombre, metrics in resultados.items():
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
        contenido += f"| {nombre} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
        contenido += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {roc_str} |\n"
    
    contenido += "## Interpretación inicial\n\n"
    contenido += "Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.\n"
    contenido += "- Accuracy indica el porcentaje general de aciertos.\n"
    contenido += "- Precision indica qué tan confiables son las predicciones positivas.\n"
    contenido += "- Recall indica qué proporción de clientes con churn fueron identificados.\n"
    contenido += "- F1-score resume precision y recall en una sola métrica.\n\n"

    contenido += "## ROC-AUC (nueva métrica)\n\n"
    contenido += "El AUC (Area Under the Curve) mide la capacidad del modelo para distinguir entre clases.\n"
    contenido += "- 0.5: modelo aleatorio\n"
    contenido += "- 0.7-0.8: aceptable\n"
    contenido += "- 0.8-0.9: excelente\n"
    contenido += "- 0.9-1.0: sobresaliente\n"

    METRICS_FILE.write_text(contenido, encoding="utf-8")

    print("\n Evaluación completada")
    print(f" Métricas guardadas en: {METRICS_FILE}")

if __name__ == "__main__":
    evaluar_modelos()
