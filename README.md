SmartMarkdown
ML-Driven Profit-Aware Retail Markdown Optimization


1. Project Overview

SmartMarkdown is a Data Science and Machine Learning project developed to support retail markdown decisions.

The objective of the project is to predict next-week product demand and recommend an appropriate markdown percentage while balancing:

- Sell-through improvement
- Inventory reduction
- Gross profit protection
- Margin constraints
- Demand uncertainty

The project separates demand prediction from markdown decision-making.

The ML model predicts future demand, while a separate decision engine evaluates multiple markdown scenarios and selects the smallest feasible discount that satisfies business constraints.


2. Business Problem

Retailers often reduce prices to move slow-moving or excess inventory.

However, deeper markdowns can:

- Increase demand
- Improve sell-through
- Reduce leftover inventory

but they can also:

- Reduce unit margin
- Reduce gross profit
- Cause unnecessary discounting

SmartMarkdown helps balance these trade-offs using historical retail data, demand forecasting, SKU-level price-response analysis, inventory information and business rules.


3. Dataset

Primary dataset:

SmartMarkdown_Data_Dev_2024_2025.xlsx

Dataset summary:

- 4,200 rows
- 30 original columns
- 40 SKUs
- 105 weekly periods
- Date range: January 2024 to December 2025
- Granularity: One record per SKU per week

Important feature groups include:

Product:
- SKU_ID
- Product_Name
- Category
- Subcategory
- Brand

Pricing:
- MRP
- Unit_Cost
- Selling_Price
- Discount_Pct
- Competitor_Price

Inventory:
- Opening_Inventory
- Replenishment_Qty
- Available_Inventory
- Closing_Inventory
- Days_In_Inventory

Demand and Sales:
- Latent_Demand
- Units_Sold
- Stockout_Flag

Promotion and Calendar:
- Promotion_Flag
- Retail_Event
- Festival_Flag
- Season_Tag

Engagement:
- Views
- Cart_Adds

Financial:
- Revenue
- Gross_Profit
- Margin_Pct


4. Project Workflow

The complete project workflow is:

Raw Retail Data
    ↓
Data Validation
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Product Clustering
    ↓
Demand Forecasting
    ↓
Price Response Analysis
    ↓
Prediction Uncertainty
    ↓
Markdown Scenario Simulation
    ↓
Business Constraint Evaluation
    ↓
Final Markdown Recommendation
    ↓
Backtesting
    ↓
Flask Application


5. Notebook Structure

The development workflow is organized into the following notebooks:

01_data_understanding_validation.ipynb
- Dataset loading
- Data validation
- Business consistency checks
- Missing values
- Duplicate checks
- Inventory and financial validation

02_eda.ipynb
- Demand analysis
- Discount analysis
- Inventory analysis
- Retail event analysis
- Correlation analysis
- SKU-level behaviour analysis

03_feature_engineering.ipynb
- Target creation
- Lag features
- Rolling demand features
- Price features
- Inventory features
- Engagement features
- Future calendar features
- Leakage prevention

04_product_clustering.ipynb
- SKU behavioural profiling
- Feature scaling
- K-Means clustering
- Cluster interpretation
- Cluster mapping generation

05_demand_regression.ipynb
- Baseline model
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Hyperparameter tuning
- Model comparison
- Final model selection

06_price_response_uncertainty.ipynb
- SKU-level discount-demand response
- Price-response normalization
- Residual analysis
- Prediction interval development
- Adaptive uncertainty framework

07_markdown_decision_engine.ipynb
- Markdown scenario generation
- Business constraint logic
- Recommendation selection
- Portfolio recommendations
- Backtesting
- Final business KPI generation


6. Forecasting Setup

The project follows a time-based forecasting structure:

Information available through Week t
    ↓
Predict Latent Demand for Week t+1

The target used by the demand model is:

Future_Latent_Demand

Latent demand is used instead of Units_Sold because observed sales may be limited by available inventory during stockout situations.


7. Feature Engineering

The final model uses 33 features.

Important engineered features include:

Demand History:
- Lag_1_Demand
- Lag_2_Demand
- Lag_4_Demand
- Rolling_4W_Avg_Demand
- Rolling_8W_Avg_Demand
- Rolling_4W_Std_Demand

Pricing:
- Discount_Pct
- Price_Index
- Competitor_Gap_Pct
- Unit_Margin

Inventory:
- Closing_Inventory
- Sell_Through
- Weeks_of_Supply
- Inventory_Pressure
- Days_In_Inventory

Engagement:
- Lag_1_Views
- Lag_1_Cart_Adds
- Rolling_4W_Avg_Views
- Rolling_4W_Avg_Cart_Adds

Future Calendar:
- Future_Month
- Future_Quarter
- Future_Week_of_Year
- Future_Festival_Flag
- Future_Retail_Event
- Future_Season_Tag

Categorical features are handled using OneHotEncoder.

Chronological data splitting and lag-based feature construction are used to prevent data leakage.


8. Product Clustering

K-Means clustering is used to segment SKUs based on behavioural characteristics.

Three clusters were selected:

1. Slow-Moving Markdown-Responsive
2. Volatile Low-Response
3. High-Demand Fast-Moving

Clustering is mainly used for:

- Product behaviour interpretation
- Portfolio understanding
- Supporting explainability

Clustering is not used as the primary demand prediction model.


9. Demand Models Tested

The following models were evaluated:

- 4-week moving average baseline
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The final selected model is:

GradientBoostingRegressor

Final model configuration:

- n_estimators = 300
- learning_rate = 0.03
- max_depth = 4
- min_samples_leaf = 1
- subsample = 1.0


10. Final Demand Model Performance

Final untouched test-set results:

MAE:
7.00

RMSE:
9.76

WAPE:
17.05%

R²:
0.737

Gradient Boosting performed better than Ridge Regression across all final test metrics and was selected as the final demand forecasting model.


11. Price Response Analysis

A SKU-level price-response relationship is estimated using historical discount and latent-demand behaviour.

For each SKU, the project calculates:

- Discount-demand slope
- Discount-demand correlation
- Average historical demand
- Relative response per 1 percentage-point discount change

The response is normalized so that high-volume SKUs do not automatically appear more price-sensitive.


12. Prediction Uncertainty

The project includes an adaptive residual-based prediction interval framework.

Demand bands:

Low Demand:
Predicted demand <= 25

Medium Demand:
Predicted demand between 26 and 50

High Demand:
Predicted demand > 50

The selected 95% adaptive prediction interval achieved approximately:

93.17% test coverage

The lower demand estimate is used for conservative profitability checks.


13. Markdown Decision Engine

The ML model does not directly predict the recommended discount.

Instead, the decision engine evaluates multiple candidate markdown scenarios.

Candidate discounts:

0%
5%
8%
10%
12%
15%
18%
20%
23%
25%
27%
30%

For each candidate markdown, the engine calculates:

- Adjusted demand
- Lower demand estimate
- Upper demand estimate
- Expected selling price
- Expected units sold
- Projected sell-through
- Revenue
- Gross profit
- Margin
- Expected leftover inventory
- Profit retention
- Conservative profit retention


14. Business Constraints

Current markdown decision rules include:

Maximum Allowed Discount:
30%

Minimum Margin:
35%

Minimum Expected Profit Retention:
80%

Minimum Conservative Profit Retention:
75%

Target Sell-Through:
45%


15. Recommendation Logic

The decision engine follows this sequence:

1. Generate candidate markdown scenarios.
2. Calculate demand and financial outcomes.
3. Remove scenarios that violate business constraints.
4. Identify feasible scenarios meeting the sell-through target.
5. Select the smallest feasible markdown that meets the target.
6. If no feasible scenario reaches the target, select the feasible scenario with the highest projected sell-through.
7. If the SKU has zero inventory, return a No Action recommendation.


16. Example Recommendation

Example SKU:

SKU003

Baseline Demand:
44.16 units

Current Discount:
0%

Recommended Discount:
15%

Decision Type:
Increase Markdown

Projected Sell-Through:
46.20%

Expected Gross Profit:
₹41,382.58

Expected Margin:
44.68%

Expected Leftover Inventory:
63.49 units

Reason:

15% is the smallest feasible markdown that meets the required sell-through target while satisfying profitability constraints.


17. Backtesting

Backtesting was performed to evaluate the complete recommendation policy on unseen historical weeks.

Backtest period:

September 2025 to December 2025

Total decisions:

600 SKU-week decisions

Actionable decisions:

585

Because some target weeks contained replenishment, a stricter no-replenishment subset was also evaluated.

Primary backtest:

490 decisions

Historical Gross Profit:
₹14,760,674.65

Simulated SmartMarkdown Gross Profit:
₹15,341,191.56

Simulated Difference:
+₹580,516.91

Simulated Improvement:
+3.93%

Important:

This result is a counterfactual simulation and should not be interpreted as guaranteed real-world business uplift.


18. Saved Model Artefacts

The models folder contains saved Joblib artefacts used by the project.

Important files include:

smartmarkdown_demand_model.joblib
- Final trained demand forecasting model

smartmarkdown_demand_model_metadata.joblib
- Model metadata
- Feature names
- Target information
- Feature count
- Evaluation details

smartmarkdown_decision_config.joblib
- Candidate discounts
- Business thresholds
- Prediction interval settings

smartmarkdown_price_response.joblib
- SKU-level price-response data

smartmarkdown_cluster_mapping.joblib
- Saved SKU cluster mapping


19. Processed Data Files

Important processed files include:

smartmarkdown_feature_engineered.csv
- Final model-ready feature dataset

smartmarkdown_price_response.csv
- SKU-level price-response analysis

smartmarkdown_sku_cluster_mapping.csv
- SKU-to-cluster mapping

smartmarkdown_validation_residuals.csv
- Validation residuals used for uncertainty analysis

smartmarkdown_test_predictions.csv
- Final demand-model test predictions


20. Application

The final application uses Flask.

Main application files:

app.py
clustering_engine.py

Templates:

templates/
- index.html
- clusters.html
- recommendations.html

Static assets:

static/
- css/
- images/

The application allows users to select an SKU and generate markdown recommendations.

Depending on the final UI configuration, optional overrides may include:

- Target week
- Current discount
- Closing inventory
- Competitor price
- Retail event
- Festival flag
- Season


21. Runtime Workflow

User
    ↓
Flask UI
    ↓
app.py
    ↓
Feature Preparation
    ↓
Saved Demand Model
    ↓
Price Response
    ↓
Prediction Interval
    ↓
Markdown Decision Engine
    ↓
Recommendation
    ↓
Flask UI Output


22. Project Structure

SmartMarkdown - Project/

├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding_validation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_product_clustering.ipynb
│   ├── 05_demand_regression.ipynb
│   ├── 06_price_response_uncertainty.ipynb
│   └── 07_markdown_decision_engine.ipynb
│
├── outputs/
│   ├── charts/
│   └── final/
│
├── src/
│   ├── __init__.py
│   ├── feature_builder.py
│   └── markdown_decision_engine.py
│
├── static/
│
├── templates/
│
├── app.py
├── clustering_engine.py
├── requirements.txt
└── readme.txt


23. Installation

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

python -m pip install --upgrade pip

python -m pip install -r requirements.txt


24. Running the Application

From the project root:

python app.py

The Flask development server will typically run at:

http://127.0.0.1:5000

Open the URL in a browser to access the application.


25. Main Python Dependencies

The project uses:

- Flask
- pandas
- NumPy
- scikit-learn
- joblib
- openpyxl
- matplotlib
- seaborn
- Jupyter


26. Important Limitations

The current project has several limitations:

- The dataset is synthetic.
- Price response is association-based and not causal price elasticity.
- The decision engine does not currently model future replenishment directly.
- Business thresholds are mostly static across SKUs.
- Only two years of weekly history are available.
- Backtesting is counterfactual simulation rather than a live A/B test.
- Final Flask runtime integration should be tested from a clean environment before release.


27. Future Improvements

Possible improvements include:

- Dynamic sell-through targets
- Category-specific business constraints
- Replenishment-aware inventory simulation
- Causal price elasticity estimation
- Longer historical datasets
- Automated retraining
- Model drift monitoring
- Unit tests
- API-based architecture
- Cloud deployment
- Model and data versioning


28. Key Technical Learnings

Important learnings from this project include:

- Time-based data should use chronological splitting.
- Lag and rolling features must be constructed carefully to avoid leakage.
- Sales and demand are not always the same when stockouts occur.
- Model prediction and business optimization should be separated.
- A strong model alone is not enough for a business decision system.
- Prediction uncertainty can improve decision safety.
- Backtesting should evaluate the full decision policy, not only model accuracy.
- Saved model artefacts, metadata and runtime feature schemas must remain synchronized.
