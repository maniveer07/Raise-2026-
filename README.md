# RAISE-26: Daily Life vs Grand Narratives

Compare **micro (daily life)** vs **macro (grand narratives)** framings of AI using:
- A labeled **news headlines** dataset (Datasets A/B)
- Persona-based **LLM outputs** (Dataset C)

This repo includes a notebook “one-click” pipeline and a small Python script for quick summary tables.

## What’s in this repo

- **`RAISE-26_micro_macro.ipynb`**: end-to-end analysis pipeline (load data → classify/aggregate → plots & tables).
- **`report_extract.py`**: prints summary tables (no files written).

## Data (expected folder + filenames)

Create a `data/` folder in the repo root and place the CSVs below inside it.

### News datasets (A or B)

- **Dataset A**: `data/dataset_A_news_full_10500.csv`
- **Dataset B**: `data/dataset_B_news_subset_3500.csv`

Required columns (names are flexible; the script normalizes and looks for common variants):
- **title/headline**
- **source/publisher/outlet**
- **date/published timestamp**
- **classes / labels / categories** as a semicolon-separated string (e.g. `"Routine, Lifestyle & Behavior; Society, Ethics & Culture"`)

### Persona + LLM outputs (Dataset C)

Either of these filenames is accepted:
- `data/Dataset_C_prompts_&_queries.csv`
- `data/Dataset_C_prompts_-_queries.csv`

Required columns (names are flexible; the script normalizes and looks for common variants):
- **Prompt**
- **Query**
- **LLM** (model name)
- **LLMoutput** (model response/completion)

## Quickstart (Windows / PowerShell)

### 1) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Install dependencies

```powershell
pip install pandas numpy matplotlib seaborn
```

If you want to run the notebook:

```powershell
pip install jupyter
```

## Run the notebook pipeline

```powershell
jupyter lab
```

Open `RAISE-26_micro_macro.ipynb` and set `USE_DATASET = "A"` (or `"B"`) near the top.

## Run the summary script

1) Open `report_extract.py` and set `USE_DATASET = "A"` (or `"B"`).
2) Run:

```powershell
python .\report_extract.py
```

It prints:
- `NEWS_LEVEL` / `PERSONA_LEVEL` (micro/macro/both/unknown distribution)
- `TOP_SOURCES_SHARE` (top outlets by level share)
- `TOP_PERSONAS_SHARE` (top personas by level share)
- `LLM_SHARE` (model-by-level share)
- `YEAR_SHARE` (micro vs macro share by year, from the news dataset)

Tip: capture output to a file:

```powershell
python .\report_extract.py | Tee-Object -FilePath .\summary.txt
```
