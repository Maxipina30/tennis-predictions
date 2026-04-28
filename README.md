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
