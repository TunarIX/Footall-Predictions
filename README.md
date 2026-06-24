# Football Predictions

Football Predictions is a football data analytics and probability estimation dashboard. It is **not a betting app**: the goal is to explore historical football data, bookmaker-implied probabilities, and transparent model estimates.

## Features

- Upload one or more historical match CSV files.
- Use a modular competition configuration in `config/competitions.yml`.
- Support major football-data.co.uk club league CSVs and separate international CSV inputs.
- Clean and standardize common match, score, result, and bookmaker odds columns.
- Analyze last-5 and last-10 form, home/away performance, goals scored and conceded trends, draw rates, and team form.
- Calculate chronological Elo ratings for teams.
- Add head-to-head features for recent meetings.
- Convert decimal bookmaker odds into normalized implied probabilities.
- Train a multi-feature baseline model that uses odds as only one input, alongside team form, venue performance, goals, H2H, and Elo.
- Estimate home win, draw, and away win probabilities for an upcoming match.
- Estimate a likely score, confidence score, and explanation notes for each prediction.
- Find similar historical matches using engineered football features as supporting context, not just similar odds.

## Project structure

```text
app.py
config/
  competitions.yml
src/
  competitions.py
  data_loader.py
  elo.py
  preprocessing.py
  features.py
  odds.py
  predictor.py
  visualization.py
requirements.txt
README.md
```

## Competition configuration

Supported competitions are configured in `config/competitions.yml`. Each entry contains:

```yaml
- name: Premier League
  country_or_type: England
  data_source: football-data.co.uk
  football_data_code: E0
  match_type: club
```

Current focus competitions are Premier League, La Liga, Serie A, Bundesliga, Ligue 1, FIFA World Cup, and general international matches.

### Add a new league later

1. Open `config/competitions.yml`.
2. Add a new item under `competitions`.
3. For football-data.co.uk leagues, set `data_source: football-data.co.uk` and add the site code when available, such as `E0`, `SP1`, `I1`, `D1`, or `F1`.
4. For national-team datasets, set `data_source: international_csv` and `match_type: international`.
5. Restart Streamlit and select the new competition in the sidebar.

No Python code change is required for a new competition that follows one of the existing loader formats.

## Input formats

### football-data.co.uk club CSVs

The app recognizes common football-data.co.uk columns including:

`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HTHG`, `HTAG`, `HTR`, `B365H`, `B365D`, `B365A`, `BWH`, `BWD`, `BWA`, `IWH`, `IWD`, `IWA`, `PSH`, `PSD`, `PSA`, `MaxH`, `MaxD`, `MaxA`, `AvgH`, `AvgD`, `AvgA`.

### International CSVs

International files can include columns such as `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `country`, `neutral`, and optional odds columns `home_odds`, `draw_odds`, `away_odds`. The loader maps these to the app's canonical format and derives `FTR` from the score when needed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Model notes

The baseline model is intentionally understandable and modular. It uses Random Forest classification over engineered pre-match features:

- last-5 and last-10 points per game;
- last-5 and last-10 goals scored/conceded;
- home-team home performance and away-team away performance;
- recent head-to-head history;
- chronological Elo ratings;
- bookmaker implied probabilities;
- similar historical matches as context.

Odds are **not** the only driver of predictions. They are treated as market context alongside football performance features.

## Responsible framing

This dashboard presents probabilities and historical analytics for education and research. A value signal is only an analytical comparison between model estimates and bookmaker implied probability; it is not financial advice or a recommendation to wager.
