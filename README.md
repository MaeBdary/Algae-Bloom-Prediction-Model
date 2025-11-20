# Algae Bloom Prediction Model

This repository contains the code and notebooks for a machine learning pipeline that predicts harmful algae bloom severity from environmental and water-quality data.

---

## Goal

Develop a supervised learning model that can reliably forecast algae bloom intensity using multi-source tabular data (e.g., water quality, weather, and temporal features). The objective is to provide a quantitative tool that supports early warning and decision-making for water resource management.

---

## Motivation & Reasoning

Harmful algae blooms pose serious risks to ecosystems, drinking water, and recreation. Traditional monitoring is:

- **Reactive** – action is taken only after blooms are observed.
- **Labor-intensive** – requires frequent sampling and lab analysis.
- **Local** – hard to generalize across regions or changing conditions.

A data-driven prediction model can:

- Anticipate elevated bloom risk before it becomes critical.
- Help prioritize sampling and interventions.
- Serve as a foundation for more advanced, real-time monitoring systems.

To achieve this, the project focuses on:

- Cleaning and aligning heterogeneous data sources.
- Engineering features that encode temporal dynamics and environmental context.
- Systematically comparing several regression models to identify the most accurate and robust approach.

---

## Repository Structure

- `Preprocessing.ipynb`  
  End-to-end preprocessing pipeline:
  - Data loading from raw sources.
  - Handling missing values and noisy measurements.
  - Feature engineering (temporal aggregates, derived variables).
  - Train/validation/test splitting and scaling.

- `model-checkpoint.ipynb`  
  Model training, evaluation, and comparison:
  - Fits multiple regression models.
  - Computes and logs performance metrics.
  - Generates plots and diagnostics.

(Additional scripts, data loaders, and configuration files can be added around these notebooks as the project grows.)

---

## Methods

The modeling pipeline includes:

1. **Data Preprocessing**
   - Cleaning and imputing missing sensor readings.
   - Scaling numerical features for linear models.
   - Constructing time-windowed statistics (e.g., rolling averages) to capture recent trends.
   - Aligning multi-source measurements on a common temporal index.

2. **Models Evaluated**
   - **ElasticNet Regression**
   - **Histogram Gradient Boosting Regressor (HGBR)**
   - **PCA + XGBoost Regressor**

3. **Evaluation Metrics**
   - Coefficient of determination: **R²**
   - Root Mean Squared Error: **RMSE**
   - Mean Absolute Error: **MAE**

---

## Results

On the held-out test set, the models achieved:

| Model     |   R²   |  RMSE  |  MAE   |
|----------|:------:|:------:|:------:|
| ElasticNet | 0.6343 | 2.5586 | 1.5321 |
| HGBR       | 0.8560 | 1.6056 | 0.8586 |
| PCA + XGB  | 0.8235 | 1.7775 | 0.9719 |

**Key takeaway:** The Histogram Gradient Boosting Regressor provides the best trade-off between accuracy and error magnitude, achieving **R² = 0.856**, **RMSE = 1.6056**, and **MAE = 0.8586**, and significantly outperforming both the ElasticNet and PCA-augmented XGBoost baselines.

---

## Impact & Future Work

This project demonstrates that modern tree-based models can accurately forecast algae bloom severity from tabular environmental data, creating a foundation for practical decision-support tools.

Planned and potential future extensions include:

- **Richer Data Sources**  
  Incorporate satellite-derived indices, higher-resolution meteorological data, and additional water-quality parameters.

- **Spatiotemporal Modeling**  
  Move beyond single-location predictions by modeling spatial correlations across multiple monitoring sites or water bodies.

- **Uncertainty Quantification**  
  Add prediction intervals or probabilistic models to communicate confidence levels alongside point estimates.

- **Deployment**  
  Package the trained model behind an API or dashboard so environmental agencies can upload recent measurements and receive bloom-risk forecasts in real time.

- **Model Monitoring**  
  Track performance over time and add automated retraining when data distribution shifts (e.g., new seasons, climate trends, or instrumentation changes).

