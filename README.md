# Proyecto Churn MLOps

Este proyecto corresponde a una práctica inicial del módulo de MLOps.

El objetivo es construir una estructura básica de trabajo para un proyecto de Machine Learning que permita:

- Preparar datos.
- Entrenar dos modelos.
- Evaluar métricas.
- Guardar los modelos entrenados.
- Exponer los modelos mediante una API.
- Ejecutar pruebas básicas.

## Problema del proyecto

Se trabajará con un caso simplificado de predicción de abandono de clientes, conocido como churn.

Los modelos intentarán predecir si un cliente podría abandonar un servicio, utilizando variables como antigüedad, saldo promedio y reclamos.

## Estructura del proyecto

```text
proyecto_churn_mlops
├── data
├── notebooks
├── src
├── models
├── api
├── tests
├── docs
├── README.md
└── requirements.txt
```

## Carpetas principales

- `data`: contiene los datos del proyecto.
- `notebooks`: contiene análisis exploratorios.
- `src`: contiene los scripts principales de los modelos.
- `models`: contiene los modelos entrenados.
- `api`: contiene la API de los modelos.
- `tests`: contiene pruebas automáticas.
- `docs`: contiene documentación y métricas.

## Flujo inicial del proyecto

El flujo básico será:

1. Entrenar los modelos.
2. Guardar las métricas.
3. Crear los modelos serializados
4. Crear una API.
5. Probar el funcionamiento inicial.
6. Validar los datos enviados a la API.

## Control de versiones

Este proyecto utiliza Git para registrar cambios y GitHub para respaldar el repositorio en la nube.
