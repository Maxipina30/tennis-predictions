# Non-Grand-Slam BO3 Models Pipeline

Pipeline especifico para torneos ATP no Grand Slam al mejor de 3 sets: Masters 1000, ATP 500 y ATP 250.

## Estructura

Codigo:

- `src/non_grand_slam_bo3/01_prepare_split.py`: filtra partidos BO3 no Grand Slam desde el dataset general, balancea lado J1/J2 y crea train/test.
- `src/non_grand_slam_bo3/02_train_moneyline.py`: entrena regresiones logisticas para moneyline y mercados de sets BO3.
- `src/non_grand_slam_bo3/03_surface_report.py`: evalua metricas por superficie y hace un holdout temporal interno para grass.
- `src/non_grand_slam_bo3/04_predict_upcoming.py`: aplica modelos BO3 entrenados sobre features upcoming.

Artefactos:

- `files/processed/non_grand_slam_bo3/model_dataset/`
- `files/processed/non_grand_slam_bo3/model_training/`

## Targets

- J1 gana partido.
- J1 gana 2-0.
- J2 gana 2-0.
- J1 gana al menos un set.
- J2 gana al menos un set.
- Over 19.5 games.

Nota: `gana al menos un set` es el complemento del 2-0 del rival, pero se entrena/expone como target para mantener el mismo estilo operativo que el pipeline Grand Slam.

## Comandos

Preparar split:

```powershell
venv312\Scripts\python.exe src\non_grand_slam_bo3\01_prepare_split.py
```

Entrenar:

```powershell
venv312\Scripts\python.exe src\non_grand_slam_bo3\02_train_moneyline.py
```

Reporte por superficie:

```powershell
venv312\Scripts\python.exe src\non_grand_slam_bo3\03_surface_report.py
```

Predecir un upcoming ya convertido a features:

```powershell
venv312\Scripts\python.exe src\non_grand_slam_bo3\04_predict_upcoming.py --upcoming files\processed\model_dataset_2025_2026\test_<torneo>_<ronda>_upcoming.csv --prediction-output <torneo>_<ronda>_predictions.csv
```

## Estado con datos actuales

Dataset BO3 filtrado:

- 2803 partidos BO3 no Grand Slam.
- 2735 train.
- 68 test, todos clay por el holdout actual Barcelona/Madrid/Munich.

Distribucion por superficie:

- hard: 1309.
- clay: 894.
- unknown: 435.
- grass: 165.

Metricas test BO3 actual:

- Moneyline: accuracy 0.662, log loss 0.577, ROC AUC 0.743.
- J1 2-0: accuracy 0.721, log loss 0.542, ROC AUC 0.709.
- J2 2-0: accuracy 0.559, log loss 0.673, ROC AUC 0.639.
- Over 19.5 games: accuracy 0.544, log loss 0.727, ROC AUC 0.417.

## Hierba

Con el dataset actual no recomiendo activar un modelo operativo solo de grass todavia.

Motivo:

- Solo hay 165 partidos grass, todos en 2025.
- No hay holdout grass 2026 en el dataset BO3 actual.
- La validacion temporal interna grass deja moneyline con ROC AUC 0.47 y log loss 0.91, peor que el modelo global BO3 en el holdout operativo.

Recomendacion:

- Usar el modelo global BO3 para grass por ahora, aprovechando features especificas de superficie.
- Mantener `03_surface_report.py` como semaforo.
- Reconsiderar modelo grass-only cuando haya mas muestra, idealmente incluyendo torneos grass 2026 completos o historico BO3 de 2022-2024.
- Si se usa grass-only antes de eso, tratarlo como experimento y no como modelo principal del dashboard.
