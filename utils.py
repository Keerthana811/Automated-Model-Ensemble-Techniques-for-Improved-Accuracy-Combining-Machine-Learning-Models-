import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib



def load_data(filepath):
    return pd.read_csv(filepath)

def get_summary(data):
    return {
        "total_sales": data["Total"].sum(),
        "avg_rating": data["Rating"].mean(),
        "num_transactions": data.shape[0],
    }

def train_models(data):
    features = data[["Quantity", "Unit price"]]
    target = data["Total"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    base_models = [("rf", rf_model), ("xgb", xgb_model)]
    stack_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

    stack_model.fit(X_train, y_train)
    joblib.dump(stack_model, "../models/stacking_model.pkl" )
    return stack_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2



def predict_total_sales(model, input_data):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    return round(prediction, 2)


