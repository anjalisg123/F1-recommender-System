# 🏎️ F1 Track & Race Recommendation System

A machine learning-powered system that predicts race excitement, classifies fan interest levels, discovers natural race groupings, and recommends similar Formula 1 races — all built on 25 seasons of historical data.

## Problem Statement

Formula 1 hosts 20+ races per season across diverse circuits worldwide. This project leverages decades of historical race data to build a **Race Recommendation System** that helps fans discover races they'll love, using multiple ML perspectives:

- **Regression** — Predict how exciting a race will be (Linear Regression, Random Forest, Stacking Ensemble)
- **Classification** — Label races as high-interest or low-interest (Logistic Regression)
- **Clustering** — Discover natural groupings of similar races (K-Means)
- **Recommendation** — Surface similar races via cosine similarity, cluster membership, and predicted excitement rankings

## Data

**Source:** [Ergast Formula 1 API](https://api.jolpi.ca/ergast/f1) — seasons 2000 through 2024.

| Dataset | Records | Key Fields |
|---------|---------|------------|
| Races | 1,173 | Name, date, circuit, country, lat/lng |
| Results | — | Driver, constructor, grid, finish position, points, status |
| Circuits | 78 | Location details |

After aggregation, the processed dataset contains **128 race-level observations** with engineered features like `points_variance`, `competitiveness_score`, `grid_variance`, and `finish_variance`.

### Target Variables

| Variable | Type | Definition |
|----------|------|------------|
| `race_excitement_score` | Continuous | `finish_variance + grid_variance + normalized_total_points` |
| `high_interest` | Binary | 1 if excitement score > median, else 0 |

## Models & Results

### Regression Models

| Model | Test MSE | Test MAE | Test R² | CV R² (5-Fold) |
|-------|----------|----------|---------|----------------|
| Linear Regression | 16.25 | 3.16 | 0.9704 | 0.8966 ± 0.04 |
| Random Forest (200 trees) | 19.34 | 1.83 | 0.9647 | 0.9579 ± 0.02 |
| **Stacking Ensemble** | **10.79** | **1.29** | **0.9803** | 0.9484 ± 0.03 |

The **Stacking Ensemble** (Linear Regression + Random Forest → Gradient Boosting meta-learner) achieves the best test performance with minimal train-test gap.

### Classification

| Metric | Test |
|--------|------|
| Accuracy | 96.15% |
| Precision (High-Interest) | 1.00 |
| Recall (High-Interest) | 0.89 |

### Clustering

K-Means with K=3 (selected via Elbow Method) reveals three natural race groupings — moderate driver count, lower excitement, and higher excitement clusters — visualized through PCA projection.

## Recommendation Strategies

1. **Content-Based Similarity** — Cosine similarity over numerical features + one-hot-encoded country
2. **Cluster-Based** — Recommend races from the same K-Means cluster
3. **Excitement Ranking** — Rank all races by ensemble-predicted excitement score

## Tech Stack

- Python
- scikit-learn (regression, classification, clustering, stacking)
- pandas / NumPy
- Matplotlib / Seaborn
- Jupyter Notebook

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/f1-race-recommendation.git
cd f1-race-recommendation

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

## Future Improvements

- **More data** — Fix API pagination to expand from 128 to 470+ races
- **Richer features** — Weather, safety cars, overtaking stats, lap telemetry (FastF1 API)
- **Hyperparameter tuning** — GridSearchCV / RandomizedSearchCV
- **Advanced ensembles** — XGBoost / LightGBM as additional base learners
- **User personalization** — Fan profiles with favorite teams, drivers, and circuit preferences

## License

This project is for academic/educational purposes.
