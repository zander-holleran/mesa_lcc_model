# Installation & Setup

## Prerequisites

- Python 3.10 or higher
- Git

## Clone the Repository

```bash
git clone https://github.com/zander-holleran/mesa_lcc_model.git
cd mesa_lcc_model
```

## Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

This installs Mesa, GeoPandas, NumPy, Pandas, SciPy, Matplotlib, Jupyter, and all other required packages.

## Prepare Data Files

The model requires two external data files:

- **Road geometry**: `data/roads/hw210_sl_and_curvs.parquet`
- **Expected vehicle counts**: `data/vehicle_counts/expected_counts_seconds.csv`

These are downloaded automatically by the first cells in any notebook via:

```python
from collect_external_data.expected_counts import get_expected_counts
from collect_external_data.road_geom import get_road_geometry

get_road_geometry()
get_expected_counts()
```

You do not need to run these manually -- the notebooks handle it.

## Launch Jupyter

```bash
jupyter lab
```

## Verify

Open `notebooks/season_run.ipynb` and run the first three cells. If data loads without error, your setup is working.

Next: follow the [First Run](first-run.md) guide for a walkthrough of the notebook.
