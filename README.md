# tennis-predictions

Base inicial para scrapear TennisExplorer y construir datasets de prediccion de tenis sin data leakage.

## Instalacion

```powershell
venv312\Scripts\python.exe -m pip install -r requirements.txt
```

Requiere Python 3.10 o superior.

## Prueba chica

```powershell
venv312\Scripts\python.exe src/01_scrape_tennisexplorer.py --url https://www.tennisexplorer.com/houston/2026/atp-men/ --max-details 3 --max-players 3
```

Salida esperada en `files/processed/tennisexplorer_houston_2026/`:

- `tournament_matches.csv`: resultados del torneo, marcador por sets, games totales, cuotas promedio H/A y links.
- `match_details.csv`: snapshot del detalle de partido: superficie, H2H, W/L por superficie antes del partido y cuotas promedio.
- `match_homeaway_bookmaker_odds.csv`: cuotas Home/Away por casa cuando estan disponibles en el detalle.
- `player_profiles.csv`: datos basicos de cada jugador del torneo.
- `player_surface_records.csv`: historial W/L por anio y superficie.
- `player_played_matches.csv`: partidos del anio visibles en el perfil del jugador.

## Temporada ATP 2026

Scrapea torneos ATP principales del calendario TennisExplorer para el anio indicado:

```powershell
venv312\Scripts\python.exe src/01_scrape_tennisexplorer.py --batch-atp --year 2026 --out files/processed/atp_2026 --levels grand_slam masters_1000 atp_500 atp_250 --delay 1.0
```

Para traer solo torneos terminados:

```powershell
venv312\Scripts\python.exe src/01_scrape_tennisexplorer.py --batch-atp --year 2026 --completed-only --out files/processed/atp_2026
```

Para testear rapido sin bajar todo:

```powershell
venv312\Scripts\python.exe src/01_scrape_tennisexplorer.py --batch-atp --year 2026 --max-tournaments 2 --max-details-per-tournament 5 --max-players 5
```

## Features sin leakage

Una vez generado `matches.csv`, construye features pre-partido usando solo partidos anteriores:

```powershell
venv312\Scripts\python.exe src/02_build_features.py --matches files/processed/atp_2026/matches.csv --details files/processed/atp_2026/match_details.csv --out files/processed/atp_2026/features_no_leakage.csv
```

Tambien puedes correr todo con:

```powershell
.\run_atp_2026_pipeline.ps1 -Year 2026 -Out files/processed/atp_2026 -Delay 1.0
```

Cuando quieras sumar 2025 + 2026, scrapea ambos anios y construye features con los dos historicos juntos:

```powershell
venv312\Scripts\python.exe src/03_build_features_many.py --matches files/processed/atp_2025/matches.csv files/processed/atp_2026/matches.csv --details files/processed/atp_2025/match_details.csv files/processed/atp_2026/match_details.csv --out files/processed/atp_2025_2026/features_no_leakage.csv
```

## Que evita leakage

`features_no_leakage.csv` se construye ordenando cronologicamente los partidos. Para cada partido calcula:

- record previo del jugador;
- record previo en la superficie del partido;
- forma reciente previa, ultimos 10 partidos;
- games/sets promedio previos;
- H2H previo entre ambos jugadores;
- snapshot pre-partido de TennisExplorer desde `match_details.csv`, incluyendo W/L por superficie e H2H de la pagina de detalle;
- cuotas y target del partido.

Despues de generar la fila, recien ahi actualiza el historial con el resultado de ese partido.

## Notas

- TennisExplorer expone bastante informacion en HTML estatico: torneo, cuotas, detalle, H2H y records por superficie.
- Conviene usar `--delay` para no hacer requests demasiado agresivos.
- El parser guarda algunos campos crudos para revisar casos raros de formato antes de usar los datos en modelos.
- El calendario se filtra para ATP main tour y excluye challengers, ITF, UTR, exhibiciones y Davis/United Cup.

## Dashboard Streamlit

```powershell
venv312\Scripts\python.exe -m streamlit run app.py
```

La app usa predicciones ya generadas y archivos livianos versionados en `files/processed/` para poder desplegar en Streamlit Community Cloud.

### Preparar otro torneo/ronda para el dashboard

El proceso completo (scrape del fixture, refresh de rankings ATP, scrape de detalle, rebuild de dataset, reentrenar y dejar `files/processed/dashboard_target.json` apuntando al nuevo objetivo) esta parametrizado por torneo y ronda.

#### Wrapper PowerShell (recomendado)

```powershell
.\run_round_pipeline.ps1 -Tournament Rome -Round 2R
```

Parametros del wrapper:

- `-Tournament` (obligatorio): nombre del torneo como aparece en TennisExplorer (`Rome`, `Madrid`, `Roland Garros`, etc).
- `-Round` (obligatorio): identificador de ronda (`1R`, `2R`, `3R`, `R16`, `QF`, `SF`, `F`).
- `-Year` (opcional, default `2026`).
- `-Label` (opcional, default `"{Tournament} {Round}"`).
- `-Url` (opcional, default `https://www.tennisexplorer.com/{slug}/{year}/atp-men/`).
- `-ReferenceDate` (opcional, default fecha de hoy en formato `yyyy-MM-dd`).
- `-RawTodayOffsetDays` / `-RawTomorrowOffsetDays`: corrigen textos relativos del sitio (`today` / `tomorrow`) cuando la hora del torneo no coincide con tu zona horaria.
- `-Delay`: segundos entre requests al scrapear (default `0.5`).
- `-SkipTournamentScrape` / `-SkipRankingRefresh`: switches para reanudar el pipeline si el fixture o los rankings ya estan al dia.
- `-LaunchDashboard`: switch para arrancar Streamlit al terminar.
- `-Python`: ruta al interprete (default `venv312\Scripts\python.exe`).

Ejemplos:

```powershell
# Roma 2R, fecha de referencia hoy, offsets +1/+2 porque la PC va atrasada respecto al sitio
.\run_round_pipeline.ps1 -Tournament Rome -Round 2R -RawTodayOffsetDays 1 -RawTomorrowOffsetDays 2

# Madrid QF para 2025, lanzando dashboard al final
.\run_round_pipeline.ps1 -Tournament Madrid -Round QF -Year 2025 -LaunchDashboard

# Reanudar Roma 3R sin re-scrapear si ya bajaste el fixture y los rankings
.\run_round_pipeline.ps1 -Tournament Rome -Round 3R -SkipTournamentScrape -SkipRankingRefresh
```

#### Llamada directa al script Python

Tambien podes llamar el orquestador directo si preferis:

```powershell
venv312\Scripts\python.exe src/13_prepare_dashboard_target.py --tournament Rome --round 2R --reference-date 2026-05-09 --raw-today-offset-days 1 --raw-tomorrow-offset-days 2
```

Si no pasas `--url` ni `--label` los deriva del torneo (`Rome` -> `https://www.tennisexplorer.com/rome/2026/atp-men/`, label `Rome 2R`). Para torneos con varias palabras en el slug (ej. `Roland Garros` -> `roland-garros`) tambien queda automatico.

## Stats de tenis desde SofaScore

El scraper de SofaScore usa Playwright para abrir SofaScore y despues consume la API JSON, igual que el scraper de futbol. Primero instala los navegadores si todavia no estan:

```powershell
venv312\Scripts\python.exe -m playwright install chromium
```

Para sumar estadisticas de servicio y devolucion como aces, doble faltas, porcentaje de primer saque, puntos ganados con primer/segundo saque y break points:

```powershell
venv312\Scripts\python.exe src/14_scrape_sofascore_tennis_stats.py --matches files/processed/atp_2026/matches.csv --out files/processed/sofascore_tennis/match_stats.csv --from-date 2026-01-01 --to-date 2026-05-05
```

Luego reconstruye el dataset. Si existe `files/processed/sofascore_tennis/match_stats.csv`, `05_build_model_dataset.py` lo toma automaticamente y crea features historicas pre-partido con prefijo `sofascore_`:

```powershell
venv312\Scripts\python.exe src/05_build_model_dataset.py --sofascore-stats files/processed/sofascore_tennis/match_stats.csv
```
