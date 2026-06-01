# Métricas de modelos de churn

## Comparativa de modelos

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Regresión Logística | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
## Interpretación inicial

Estas métricas permiten evaluar el desempeño inicial del modelo de clasificación.
- Accuracy indica el porcentaje general de aciertos.
- Precision indica qué tan confiables son las predicciones positivas.
- Recall indica qué proporción de clientes con churn fueron identificados.
- F1-score resume precision y recall en una sola métrica.

## ROC-AUC (nueva métrica)

El AUC (Area Under the Curve) mide la capacidad del modelo para distinguir entre clases.
- 0.5: modelo aleatorio
- 0.7-0.8: aceptable
- 0.8-0.9: excelente
- 0.9-1.0: sobresaliente
