# Customer Segmentation for Marketing Targeting

> **Commercial-grade customer segmentation project**  
> RFM analysis + behavioral clustering for targeted marketing campaigns

---

## Team

| Role | Name | GitHub |
|------|------|--------|
| Team Lead | Stefania | [https://github.com/steffiprog] |
| ML Engineer | Anna | [ url ] |
| Data Engineer | Albina | [url] |

---

## Business Goal

Develop **interpretable customer segments** based on:
- Purchase history
- Website behavior
- Demographic data

→ Enable **personalized marketing**, improve ROI, reduce churn.

---

## Data Sources

| File | Description |
|------|-------------|
| `purchases.csv` | Transaction history (date, amount, frequency) |
| `behavior.csv` | Page views, sessions, click events |
| `demographics.csv` | Age, location, device type |

---

## 🧪 Technical Pipeline

Raw Data → Merge → Feature Engineering → Clustering → Segment Profiling → Marketing Report


### Modules

| Module | Responsibility |
|--------|----------------|
| `src/data` | Load, merge, clean data |
| `src/features` | RFM, behavioral features |
| `src/models` | K-Means, evaluation, prediction |
| `src/visualization` | Cluster plots, segment profiles |
| `config/` | Parameters, logging |
| `main.py` | End-to-end pipeline |

---

📊 Results

Elbow method → optimal number of clusters
2D/3D cluster visualizations
Segment profiles with marketing recommendations
→ See reports/marketing_strategies.md

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/customer_segmentation.git
cd customer_segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python main.py

# 4. Explore notebooks
jupyter notebook notebooks/


