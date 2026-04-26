# Predictive Quality Control: EWMA vs. Machine Learning
### NASA C-MAPSS FD001 & FD003 — With Interactive Streamlit Dashboard

## The Core Finding
| | FD001 (1 Fault Mode) | FD003 (2 Fault Modes) |
|---|---|---|
| EWMA Control Chart | ✅ Detects all failures | ❌ Misses Fault Mode 2 (HPT) |
| XGBoost ML Model | ✅ Detects all failures | ✅ Detects both fault modes |

## Dataset Facts
| Property | FD001 | FD003 |
|---|---|---|
| Operating Conditions | 1 | 1 |
| Fault Modes | 1 (HPC only) | 2 (HPC + HPT) |
| Training Engines | 100 | 100 |

## Dataset Setup
Place these files in the `dataset/` folder:
- train_FD001.txt, test_FD001.txt, RUL_FD001.txt
- train_FD003.txt, test_FD003.txt, RUL_FD003.txt

Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## Installation
```bash
pip install -r requirements.txt
```

## How to Run

### Full CLI Pipeline (trains models + saves outputs)
```bash
python main.py
```

### Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Dashboard Pages
| Page | Description |
|---|---|
| Dataset Overview | Shapes, distributions, engine lifetimes |
| Sensor Explorer | Interactive sensor trend viewer |
| EWMA Analysis | Tune λ and init_window interactively |
| ML Model | Train, evaluate, SHAP importance |
| Comparison | EWMA vs ML side by side |
| Business Value | Cost-benefit analysis |

## Project Structure
```text
predictive-quality-control/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── train_FD003.txt
│   ├── test_FD003.txt
│   └── RUL_FD003.txt
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── spc.py
│   ├── modeling.py
│   ├── explainability.py
│   └── utils.py
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   └── pages/
│       ├── __init__.py
│       ├── 01_dataset_overview.py
│       ├── 02_sensor_explorer.py
│       ├── 03_ewma_analysis.py
│       ├── 04_ml_model.py
│       ├── 05_comparison.py
│       └── 06_business_value.py
├── outputs/
│   ├── plots/
│   ├── models/
│   └── reports/
├── main.py
└── notebooks/
    └── analysis.ipynb
```

## The Math
**EWMA Recursion:**

$$Z_t = \lambda x_t + (1 - \lambda)Z_{t-1}$$

**Control Limits:**

$$UCL/LCL = \mu \pm 3\sigma\sqrt{\frac{\lambda}{2-\lambda}}$$

## Tech Stack
pandas | numpy | matplotlib | seaborn | scikit-learn | xgboost | shap | streamlit | plotly | joblib
