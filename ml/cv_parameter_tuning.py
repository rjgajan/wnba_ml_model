import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
import numpy as np

# Custom RMSE scorer (compatible with all sklearn versions)
def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

rmse_scorer = make_scorer(rmse, greater_is_better=False)

def tune_model(df):
    # Features and target
    features = ['rest_diff', 'OPP_PACE', 'avg_prev_5',
                'avg_prev_15', 'HOME_AWAY', 'OPP_DEF_RATING', 'team_rest_days']
    target = 'PTS'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # Preprocessor: One-hot encode HOME_AWAY
    preprocessor = ColumnTransformer(
        transformers=[
            ('home_away_enc', OneHotEncoder(handle_unknown='ignore'), ['HOME_AWAY'])
        ],
        remainder='passthrough'
    )

    # Pipeline: Preprocessor + XGBRegressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])

    # Hyperparameter grid for tuning
    param_grid = {
        'regressor__n_estimators': [200, 400, 600],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 4, 5],
        'regressor__subsample': [0.7, 0.9, 1.0],
        'regressor__colsample_bytree': [0.7, 0.9, 1.0]
    }

    # Cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search with custom RMSE scorer
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=rmse_scorer,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    # Fit the model with grid search
    grid_search.fit(X, y)

    # Print best params and RMSE
    print("\n✅ Best Parameters:")
    print(grid_search.best_params_)

    print(f"✅ Best Cross-Validated RMSE: {abs(grid_search.best_score_):.4f}\n")

    # Get top 5 parameter sets sorted by RMSE
    results = pd.DataFrame(grid_search.cv_results_)
    results['mean_rmse'] = abs(results['mean_test_score'])
    top5 = results.sort_values('mean_rmse').head(5)[
        ['mean_rmse', 'param_regressor__n_estimators', 'param_regressor__learning_rate',
         'param_regressor__max_depth', 'param_regressor__subsample', 'param_regressor__colsample_bytree']
    ]
    print("✅ Top 5 Parameter Combinations:")
    print(top5)

    return grid_search.best_estimator_

# Main execution
if __name__ == "__main__":
    df = pd.read_csv('processed_2024_2025.csv')  # Load your data
    best_model = tune_model(df)