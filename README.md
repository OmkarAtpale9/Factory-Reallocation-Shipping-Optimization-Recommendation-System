# Factory-to-Customer Shipping Route Efficiency Analysis

A production-ready logistics intelligence system that ingests shipping/order CSV data, engineers route and lead-time features, analyzes efficiency and bottlenecks, trains a delay prediction model, and presents insights via a Streamlit dashboard.

## What this project does

- **Data processing**: Load CSV, parse dates, drop invalid records (Ship Date < Order Date), handle missing values.
- **Feature engineering**:
  - Lead Time = Ship Date − Order Date (days)
  - Route = Factory → State/Region
  - Delay flag = Lead Time > threshold (configurable)
- **Analytics**:
  - Route-level aggregation (mean, std, volume)
  - Top best/worst routes
  - State/region bottleneck analysis
  - Ship mode comparison
- **KPIs**: Average lead time, delay frequency, route efficiency score (normalized).
- **Machine learning**: RandomForest classifier to predict delays + accuracy and confusion matrix.
- **Dashboard**: Streamlit multi-page app with filters and Plotly visualizations.
- **API-ready**: FastAPI skeleton that can serve metrics/model in the future.

## Expected CSV schema

The pipeline is robust to extra columns. The following columns are expected (case-sensitive by default; configurable):

- `Order Date` (date/datetime)
- `Ship Date` (date/datetime)
- `Factory` (string)
- `State` (string) or `Region` (string) (at least one is required)
- `Ship Mode` (string)
- `Sales` (numeric, optional but used for ML if present)
- `Units` (numeric, optional but used for ML if present)

If your data uses different names, update `config/settings.yaml` under `schema`.

## Setup

### Fresh machine setup (Windows PowerShell)

Run the commands below **from this folder** (project root).

#### 1) Open PowerShell in the project folder

```powershell
cd "e:\TY project\intership"
```

#### 2) Create a virtual environment

```powershell
python -m venv .venv
```

#### 3) Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### 4) Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

#### 5) Install dependencies

```powershell
pip install -r requirements.txt
```

### Alternative (macOS/Linux)

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit dashboard

Put your dataset in `data/` (e.g. `data/shipments.csv`) or upload it from the UI.

```bash
python -m streamlit run app/main.py
```

Then open:

- `http://localhost:8501`

To stop Streamlit, press `Ctrl + C` in the terminal.

## Optional: Run the API skeleton

```bash
uvicorn api.main:app --reload
```

API endpoints:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/config`

To stop the API, press `Ctrl + C` in the terminal.

## Project structure

```text
project/
│── data/
│── notebooks/
│── src/
│   │── data_preprocessing.py
│   │── feature_engineering.py
│   │── analysis.py
│   │── model.py
│   │── utils.py
│── app/
│   │── main.py
│   │── components/
│── api/
│── config/
│── requirements.txt
│── README.md
```

## Configuration

All configurable items live in `config/settings.yaml`:

- Column names (schema mapping)
- Delay threshold (days)
- Default data path
- Model parameters
- Logging settings

## Notes on scalability / production readiness

- **Stateless functions** in `src/` for composability and testability
- **Typed interfaces** where valuable; consistent DataFrame contracts
- **Logging** with a single configuration entrypoint
- **Model pipeline** uses scikit-learn preprocessing + joblib artifacts
- **API placeholder** designed for future integration (e.g., PostgreSQL storage, scheduled training, model registry)

## Future improvements

- Add automated tests (`pytest`) and CI
- Add database ingestion (PostgreSQL + SQLAlchemy) and incremental loads
- Add Great Expectations / pandera validation
- Add drift monitoring and periodic retraining
- Add geocoding (if latitude/longitude fields aren’t provided) for richer maps

