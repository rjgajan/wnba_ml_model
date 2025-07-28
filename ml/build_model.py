import pandas as pd
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
    return model

# main loop
if __name__ == "__main__":
    df = pd.read_csv('processed_2024_2025.csv')  # read data
    model = preprocess_and_train(df)

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