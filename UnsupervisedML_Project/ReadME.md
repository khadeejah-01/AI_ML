# ClusterCart: E-commerce Customer Segmentation System

An unsupervised machine learning project that groups e-commerce customers into meaningful segments using their demographics, spending, and purchase behaviour, so marketing and retention strategies can be personalised instead of generic.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-orange) ![Status](https://img.shields.io/badge/status-complete-green)

---

## Problem Statement

ClusterCart is a growing e-commerce platform with customers across multiple countries. It currently runs **generic marketing campaigns** for everyone, which leads to inefficient spend, missed chances to retain high-value customers, and late detection of churn-prone users.

**Goal:** build a customer segmentation system with clustering algorithms that discovers hidden behavioural patterns and supports data-driven, personalised marketing.

## Dataset

- **2,240 customers × 22 attributes** (one row per customer)
- The CSV used by the notebook is named `smartcart_customers.csv` and was obtained from Kaggle.

| Group | Features |
|---|---|
| Demographics | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer` |
| Spend (amount) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase frequency | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Feedback | `Recency`, `Complain`, `Response` |

## Pipeline

1. **Missing values:** 24 missing `Income` values filled with the **median** (robust to extreme incomes).
2. **Feature engineering:** `Age`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Children`, `Living_With`; `Education` collapsed into 3 levels.
3. **Column dropping:** removed IDs, raw/redundant columns, and the six individual spend columns (merged into `Total_Spending`).
4. **Outlier handling:** pair plots to inspect, then removed `Age ≥ 90` and `Income ≥ 600,000` (2,240 → 2,236 rows).
5. **Correlation heatmap:** on numeric features only.
6. **Encoding:** One-Hot Encoding of `Living_With` and `Education`.
7. **Scaling:** `StandardScaler` on 18 features.
8. **Dimensionality reduction:** PCA to 3 components (≈45.9% variance) for clustering and 3D visualisation.
9. **Choosing k:** Elbow method (WCSS) + Silhouette score → **k = 4**.
10. **Clustering:** K-Means, Agglomerative (Ward), and DBSCAN compared → **Agglomerative, k = 4** selected.
11. **Cluster profiling:** mean of every feature per cluster + income-vs-spending scatter plot.

## Results: The 4 Customer Segments

| Cluster | Colour | Income (avg) | Total spend (avg) | Living | Profile |
|---|---|---|---|---|---|
| 0 | 🔴 Red | ~39.7K | ~222 | With partner | Budget shoppers, deal-driven, browse the website often but buy little |
| 1 | 🔵 Blue | ~72.8K | ~1,237 | With partner | High-value, catalog + store buyers, fewer kids |
| 2 | 🟡 Yellow | ~37.0K | ~166 | Alone | Lowest spenders, most web visits, deal-driven |
| 3 | 🟢 Green | ~70.7K | ~1,190 | Alone | High-value, **most responsive to campaigns (32%)** |

**Key takeaways**
- Income and total spending are the most strongly related features and cleanly separate the customers into **high-value vs low-value** groups.
- High-income clusters buy far more through catalogs and stores; low-income clusters visit the website more but buy less and rely more on discounts.
- Cluster 3 responds to campaigns at ~32%, roughly double Cluster 1 and 4× Cluster 0, making it the best target for premium campaigns.
- Low-value clusters (0 and 2) have more children at home and higher deal usage, so discount- and bundle-based offers suit them.

> Full methodology, debugging log, and reasoning are in [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Tech Stack

`Python` · `pandas` · `NumPy` · `matplotlib` · `seaborn` · `scikit-learn` (StandardScaler, OneHotEncoder, PCA, KMeans, AgglomerativeClustering, silhouette_score) · `kneed` · `Jupyter`

## Getting Started

```bash
git clone https://github.com/khadeejah-01/AI_ML.git
cd AI_ML/UnsupervisedML_Project
```
Create a virtual environment
Create an isolated Python environment for the project:

```python -m venv .venv```

Activate it:

Windows:

```.venv\Scripts\activate```

Linux / macOS:

```source .venv/bin/activate```

Install dependencies
```bash
pip install -r requirements.txt
jupyter lab
```

Open `ClusterCart_E-commerce_Customer_Segmentation_System.ipynb` and run all cells (place `smartcart_customers.csv` in the same folder).

## Project Structure

```
├── ReadME.md
├── DOCUMENTATION.md
├── ClusterCart_E-commerce_Customer_Segmentation_System.ipynb
├── data/ readMe.md
└── images/            # plots: elbow/silhouette, 3D PCA, cluster scatter
```

## Limitations & Future Work

- Clusters 0 vs 2 and 1 vs 3 differ mainly by `Living_With`, since the one-hot features carry weight after scaling. Re-running **without** them would likely give segments driven purely by value and behaviour.
- PCA with 3 components keeps only ~46% of variance.
- Add cluster-quality metrics (Davies-Bouldin, Calinski-Harabasz) and compare alternatives such as Gaussian Mixture Models.
- Wrap the pipeline in an sklearn `Pipeline` and build a small Streamlit app that assigns new customers to a segment.

## Author

**Khadeejah**: BS Computer Science student, self-studying AI/ML.
