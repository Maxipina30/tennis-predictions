# Grand Slam Models Pipeline

Este README documenta el pipeline especifico de modelos BO5 de Grand Slams. No reemplaza el README principal del repo: su objetivo es que la proxima actualizacion de Roland Garros, Wimbledon, US Open o Australian Open sea rapida y no termine en parches manuales.

## Resumen del pipeline

El dashboard usa dos capas:

1. Datos y features pre-partido, construidos con el pipeline general del repo.
2. Modelos especificos de Grand Slam BO5, entrenados en `src/grand_slam_moneyline/`.

El modelo activo es una regresion logistica para partidos al mejor de 5 sets. Entrena con Grand Slams 2022-2025 y testea contra Australian Open 2026.

Targets entrenados:

- J1 gana partido.
- J1 gana 3-0.
- J2 gana 3-0.
- J1 -1.5 sets.
- J2 -1.5 sets.
- J1 gana al menos 2 sets.
- J2 gana al menos 2 sets.
- Over 3.5 sets, entrenado pero excluido del dashboard operativo por mala metrica.

## Estructura de carpetas

### Codigo

`src/grand_slam_moneyline/`

- `00_combine_histories.py`: combina historiales de jugadores desde varias carpetas y elimina duplicados.
- `01_prepare_split.py`: filtra partidos Grand Slam, arma `train.csv` y `test_australian_open_2026.csv`, y orienta filas de forma balanceada para evitar sesgo por lado J1/J2.
- `02_train_moneyline.py`: entrena los modelos logisticos BO5 para todos los targets y guarda metricas, modelos e importancia de features.
- `03_null_report.py`: genera reporte de nulos del dataset Grand Slam.
- `04_predict_upcoming.py`: aplica los modelos Grand Slam entrenados sobre un CSV de features de partidos upcoming.

### Datos procesados

`files/processed/grand_slam_moneyline/`

- `raw/`: scrapes base de torneos Grand Slam terminados por anio.
  - Ejemplo: `raw/grand_slam_2022/`, `raw/grand_slam_2023/`, `raw/grand_slam_2024/`.
  - Cada carpeta contiene partidos crudos y `scrape_summary.json`.
- `histories_2022_2023/`: historiales antiguos de jugadores usados para cubrir Grand Slams 2022-2023.
- `histories_combined/`: historiales combinados y deduplicados. Es la fuente limpia para features historicas.
- `sofascore/`: stats de SofaScore para partidos Grand Slam.
  - `match_stats.csv`: stats encontradas.
  - `match_stats_misses.csv`: partidos sin stats.
  - `cache/`: cache de respuestas/API.
- `model_dataset/`: dataset final para entrenar/testear.
  - `model_dataset.csv`: features completas para modelado.
  - `grand_slam_matches.csv`: partidos Grand Slam filtrados.
  - `train.csv`: train Grand Slam 2022-2025.
  - `test_australian_open_2026.csv`: test holdout.
  - `split_summary.json`: resumen de split.
  - `null_report.csv` y `null_summary.json`: diagnostico de nulos.
- `model_training/`: modelos y predicciones.
  - `metrics.csv`: metricas por target.
  - `feature_columns.json`: columnas usadas por los modelos.
  - `regresion_logistica_*.joblib`: modelos entrenados.
  - `logistic_feature_importance_*.csv`: importancia por target.
  - `test_predictions.csv`: predicciones del test.
  - `roland_garros_1r_predictions.csv`: predicciones actuales del dashboard.

### Datos del dashboard activo

`files/processed/roland_garros_1r/`

- `upcoming_matches.csv`: fixture scrapeado desde TennisExplorer. Si un partido ya se jugo, TennisExplorer puede sacarlo de este archivo.
- `upcoming_match_details.csv`: cuotas de mercados adicionales por partido.
- `match_results.csv`: resultados ya jugados y snapshot de cuotas originales para no perderlas cuando salen de upcoming.
- `tournament_metadata.json`: metadata del torneo.

`files/processed/dashboard_target.json`

Apunta el dashboard al torneo/ronda activo:

- label del dashboard;
- torneo y ronda;
- archivo de predicciones;
- archivo de upcoming;
- fecha de referencia para interpretar `today` y `tomorrow`.

## Flujo rapido para actualizar una ronda Grand Slam

Este es el flujo que conviene seguir para actualizar el dashboard sin parches manuales.

### 1. Scrape fixture de la ronda

Ejemplo Roland Garros ATP 2026:

```powershell
venv312\Scripts\python.exe src\01_scrape_tennisexplorer.py --url https://www.tennisexplorer.com/french-open/2026/atp-men/ --out files\processed\roland_garros_1r --delay 0.5 --no-details --no-players
```

Notas:

- TennisExplorer usa slug `french-open`, aunque el dashboard lo muestra como Roland Garros.
- Despues de jugarse partidos, `upcoming_matches.csv` ya no contiene todos los partidos originales.
- Por eso los resultados y cuotas de partidos jugados deben conservarse en `match_results.csv`.

### 2. Scrape detalle de cuotas para los partidos pendientes

```powershell
venv312\Scripts\python.exe src\09_scrape_upcoming_details.py --upcoming files\processed\roland_garros_1r\upcoming_matches.csv --out files\processed\roland_garros_1r\upcoming_match_details.csv --delay 0.8
```

Esto alimenta mercados como gana set, 3-0, handicap sets y 2+ sets.

### 3. Crear features upcoming

Actualmente este paso reutiliza el pipeline general. Para Roland Garros 1R, el archivo usado fue:

```text
files/processed/model_dataset_2025_2026/test_roland_garros_1r_upcoming.csv
```

Si cambia la ronda o torneo, hay que generar un nuevo `test_<torneo>_<ronda>_upcoming.csv` con `src/06_split_train_test.py`, usando:

- `files/processed/grand_slam_moneyline/model_dataset/model_dataset.csv` o el dataset general actualizado;
- historiales combinados;
- rankings ATP;
- stats SofaScore si existen;
- `--upcoming` apuntando al fixture de la ronda.

Este es el punto menos ordenado hoy: falta un wrapper especifico Grand Slam que haga este paso sin recordar tantos argumentos.

### 4. Generar predicciones con modelos Grand Slam

```powershell
venv312\Scripts\python.exe src\grand_slam_moneyline\04_predict_upcoming.py --upcoming files\processed\model_dataset_2025_2026\test_roland_garros_1r_upcoming.csv --prediction-output roland_garros_1r_predictions.csv
```

Salida:

```text
files/processed/grand_slam_moneyline/model_training/roland_garros_1r_predictions.csv
```

### 5. Actualizar dashboard target

Editar `files/processed/dashboard_target.json`:

```json
{
  "label": "Roland Garros 1R",
  "tournament": "French Open",
  "round": "1R",
  "predictions_file": "files/processed/grand_slam_moneyline/model_training/roland_garros_1r_predictions.csv",
  "upcoming_file": "files/processed/roland_garros_1r/upcoming_matches.csv",
  "local_reference_date": "2026-05-25",
  "raw_today_offset_days": 0,
  "raw_tomorrow_offset_days": 1,
  "rankings_file": "files/processed/atp_rankings/player_ranking_history.csv"
}
```

La fecha de referencia es importante porque TennisExplorer escribe horarios como `today` y `tomorrow`.

### 6. Actualizar resultados jugados

Cuando ya se jugaron partidos:

1. Agregar filas a `files/processed/<torneo_ronda>/match_results.csv`.
2. Incluir:
   - `match_id`;
   - jugadores;
   - ganador;
   - fecha;
   - score;
   - sets de J1/J2;
   - cuotas originales `odds1_avg` y `odds2_avg`;
   - `retired`;
   - fuente.

Importante: conservar `odds1_avg` y `odds2_avg` en `match_results.csv`. Si no, las cuotas desaparecen de partidos jugados porque ya no estan en `upcoming_matches.csv`.

## Reentrenar modelos Grand Slam desde cero

Solo hace falta si cambian datos historicos, rankings, SofaScore o features.

### 1. Scrapes historicos Grand Slam

Ejemplo por anio:

```powershell
venv312\Scripts\python.exe src\01_scrape_tennisexplorer.py --batch-atp --year 2024 --levels grand_slam --completed-only --out files\processed\grand_slam_moneyline\raw\grand_slam_2024 --delay 1.0 --no-details --no-players
```

### 2. Combinar historiales

```powershell
venv312\Scripts\python.exe src\grand_slam_moneyline\00_combine_histories.py --history-dirs files\processed\grand_slam_moneyline\histories_2022_2023 files\processed\player_histories_2024_2026_extended --out-dir files\processed\grand_slam_moneyline\histories_combined
```

### 3. Reconstruir dataset de modelo

Este paso reutiliza `src/05_build_model_dataset.py`. La idea es generar:

```text
files/processed/grand_slam_moneyline/model_dataset/model_dataset.csv
```

con historiales, lesiones, rankings y SofaScore actualizados.

### 4. Preparar split Grand Slam

```powershell
venv312\Scripts\python.exe src\grand_slam_moneyline\01_prepare_split.py --matches files\processed\grand_slam_moneyline\raw\grand_slam_2022\matches.csv files\processed\grand_slam_moneyline\raw\grand_slam_2023\matches.csv files\processed\grand_slam_moneyline\raw\grand_slam_2024\matches.csv files\processed\atp_2026\matches.csv --model-dataset files\processed\grand_slam_moneyline\model_dataset\model_dataset.csv --out-dir files\processed\grand_slam_moneyline\model_dataset
```

### 5. Revisar nulos

```powershell
venv312\Scripts\python.exe src\grand_slam_moneyline\03_null_report.py
```

### 6. Entrenar

```powershell
venv312\Scripts\python.exe src\grand_slam_moneyline\02_train_moneyline.py
```

Revisar:

```text
files/processed/grand_slam_moneyline/model_training/metrics.csv
```

## Estado de orden del pipeline

Lo ordenado:

- Los modelos Grand Slam estan aislados en `src/grand_slam_moneyline/`.
- Los artefactos del modelo estan agrupados bajo `files/processed/grand_slam_moneyline/`.
- El dashboard ya apunta a `grand_slam_moneyline/model_training` para predicciones BO5.
- `match_results.csv` evita perder resultados y cuotas cuando TennisExplorer saca partidos jugados del upcoming.

Lo que todavia esta medio mezclado:

- La generacion de features upcoming sigue usando scripts generales (`06_split_train_test.py`) y carpetas generales (`model_dataset_2025_2026`).
- No existe aun un `run_grand_slam_round_pipeline.ps1` que haga scrape, details, features, predict y dashboard target en un solo comando.
- Los scrapes historicos raw cubren 2022-2024 en `grand_slam_moneyline/raw`, mientras parte de 2025-2026 vive en carpetas generales ATP.

Recomendacion para la proxima limpieza:

- Crear `run_grand_slam_round_pipeline.ps1`.
- Crear `src/grand_slam_moneyline/05_prepare_upcoming_features.py` o adaptar un wrapper que llame `06_split_train_test.py` con argumentos predefinidos.
- Mover outputs upcoming Grand Slam desde `files/processed/model_dataset_2025_2026/` hacia `files/processed/grand_slam_moneyline/model_dataset/upcoming/`.
- Mantener `match_results.csv` como fuente oficial de resultados jugados del dashboard.

## Checklist de actualizacion rapida

1. Scrape fixture con `01_scrape_tennisexplorer.py`.
2. Scrape details con `09_scrape_upcoming_details.py`.
3. Generar features upcoming.
4. Correr `04_predict_upcoming.py`.
5. Actualizar `dashboard_target.json`.
6. Si ya hubo partidos jugados, actualizar `match_results.csv` con resultados y cuotas.
7. Abrir dashboard y revisar:
   - `Resumen`: fechas, cuotas, resultado pick.
   - `Mercados`: filtros por dia y partido.
   - `Datos`: metricas Grand Slam.
