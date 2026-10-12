# Predicción de diabetes con Logistic Regression y MLflow

## Descripción

Proyecto de experimentación con un modelo de regresión logística para predecir diabetes a partir de variables clínicas y demográficas. Se utiliza MLflow Tracking para registrar hiperparámetros, métricas, artefactos y el modelo entrenado.

## Tecnologías

- Python
- pandas
- scikit-learn
- MLflow
- matplotlib

## Instalación

Crear y activar un entorno virtual e instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Ejecutar el modelo con los parámetros predeterminados:

```bash
python train.py
```

Ejemplo de ejecución parametrizada:

```bash
python train.py --c 0.1 --max-iter 1000 --class-weight none --seed 42
```

El archivo CSV debe encontrarse en la ruta configurada en el argumento `--data`, cuyo valor predeterminado es `dataset/diabetes_prediction_dataset.csv`.

## Visualización de experimentos

Para abrir la interfaz local de MLflow:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Después, acceder a http://localhost:5000 y abrir el experimento **Prediccion de diabetes**.

## Métricas

Se registran accuracy, precision, recall y F1-score, además de los parámetros de cada ejecución y sus artefactos de evaluación.