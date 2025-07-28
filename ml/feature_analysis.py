import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

def preprocess_and_train(df):
    # filter for required features
    features = ['rest_diff', 'OPP_PACE', 'avg_prev_5', 
                'avg_prev_15', 'HOME_AWAY', 'OPP_DEF_RATING', 'team_rest_days']
    target = 'PTS'
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # one-hot encode home/away
    preprocessor = ColumnTransformer(
        transformers=[
            ('home_away_enc', OneHotEncoder(handle_unknown='ignore'), ['HOME_AWAY'])
        ],
        remainder='passthrough'
    )

    # ml pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    model.fit(X_train, y_train)

    # model evaluation
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Test RMSE: {rmse:.2f}")

    # feature importance analysis
    regressor = model.named_steps['regressor']

    # get feature names after one-hot encoding
    ohe = model.named_steps['preprocessor'].named_transformers_['home_away_enc']
    encoded_features = list(ohe.get_feature_names_out(['HOME_AWAY']))
    all_features = encoded_features + ['rest_diff', 'OPP_PACE', 'avg_prev_5', 'avg_prev_15', 'OPP_DEF_RATING', 'team_rest_days']

    importances = regressor.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop Features by Importance:")
    print(feature_importance_df)

    # plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    plt.show()

    return model, feature_importance_df

# main loop
if __name__ == "__main__":
    df = pd.read_csv('processed_2024_2025.csv')  # read data
    model, feature_importance_df = preprocess_and_train(df)

    # sample inputs
    example_input = pd.DataFrame([{
        'OPP_DEF_RATING': 99.4,
        'avg_prev_15': 22.27,
        'avg_prev_5': 24.2,
        'HOME_AWAY': 'Away',
        'rest_diff': 0,
        'OPP_PACE': 80,
        'team_rest_days': 5
    }])
    prediction = model.predict(example_input)
    print(f"Predicted next game points: {prediction[0]:.2f}")