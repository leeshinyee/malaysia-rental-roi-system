from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from xgboost import XGBRegressor

drop_columns = [
    'ads_id','prop_name','completion_year','monthly_rent',
    'facilities','additional_facilities','property_type_grouped',
]

models = {
    #tree model
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "LightGBM": LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=1
    ),
    #instance-based model
    "KNN": KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        metric='euclidean'
    ),
    #kernel model
    "SVR": SVR(
        kernel='rbf',
        C=10,
        gamma='scale'
    ),
    #neural network
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42
    ),
    #linear model
    "PCR": Pipeline([
        ('pca', PCA(n_components=15)),
        ('lr', LinearRegression())
    ]),
    "Polynomial": Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('lr', LinearRegression())
    ]),
    "MultivariateRegression": LinearRegression(),

    "LSSVM": SVR(
        kernel='linear',
        C=1.0
    ),
}

#load dataset
def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=drop_columns)
    y = df['monthly_rent']
    return X, y


results = []

X, y = load_xy(r"C:\Users\Shin Yee\OneDrive\FYP\final_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for model_name, model in models.items():
    print(f"\nTraining model: {model_name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #metrics evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

    print(f"Model: {model_name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    })

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

results_df = pd.DataFrame(results)
print("\n================ Result Summary ================")
print(results_df.sort_values(by="r2", ascending=False))



