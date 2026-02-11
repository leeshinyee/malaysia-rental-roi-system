import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import shap

# load dataframe
df = pd.read_csv(r"final_dataset.csv")

# useless column for train x
drop_columns = [
    'ads_id','prop_name','completion_year','monthly_rent',
    'facilities','additional_facilities','property_type_grouped',
]

X = df.drop(columns=drop_columns)
y = df['monthly_rent']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define model
lgbm_regressor = LGBMRegressor(random_state=42)

parameter = {
    'num_leaves': [30, 50, 70],
    'max_depth': [ 10, 20],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [  700, 800, 900],
    'subsample': [0.7,0.8, 0.9 ],
    'colsample_bytree': [0.6, 0.7, 0.8]
}

# random search + cross-validation
random_search = RandomizedSearchCV(
    estimator=lgbm_regressor,
    param_distributions=parameter,
    n_iter=20,        # how many combinations to try
    cv=5,             # 5-fold cross-validation
    scoring='neg_root_mean_squared_error',  # RMSE, lower = better
    random_state=42,
    n_jobs=1         # use all cores
)

# model training
random_search.fit(X_train, y_train)

print("Best Parameters Found:", random_search.best_params_)
best_model = random_search.best_estimator_

# model evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

print("\nTuned LightGBM Regressor ")
print(f"  MAE  : {mae:.2f}")
print(f"  MSE  : {mse:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f}%")

# # save model
# joblib.dump(best_model, "best_lightgbm_model.pkl")
# print("Model saved")

model_name= "lightBGM"
output_dir = os.path.dirname(os.path.abspath(__file__))

## Actual vs Predicted scatter plot
plt.figure(figsize=(10, 7))
# x = actual rent, y = model predicted rent
plt.scatter(y_test, y_pred, alpha=0.4, s=20, color="#1f77b4")
# adjust x range for better visualization
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value], [min_value, max_value],
         'r--', linewidth=2, label="Perfect Prediction")

plt.xlabel("Actual Monthly Rent", fontsize=12)
plt.ylabel("Predicted Monthly Rent", fontsize=12)
plt.title("Actual vs Predicted Monthly Rent - LightGBM", fontsize=14)
#label
plt.text(min_value, max_value,
         f"R² = {r2:.3f}\nRMSE = {rmse:.1f}\nMAE = {mae:.1f}",
         fontsize=12,
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_actual_vs_predicted.png"),dpi=300)
plt.show()

## Residual scatter plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 7))
x_jitter = y_pred + np.random.normal(0, 25, size=len(y_pred))

plt.scatter(x_jitter, residuals,
            alpha=0.15, s=12, color="#1f77b4")

plt.axhline(0, color='red', linestyle='--', linewidth=1)

plt.xlabel("Predicted Monthly Rent")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - LightGBM")

# limit y range for better visualization
plt.ylim(-800, 800)

plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_residual_plot.png"),dpi=300)
plt.show()

## Learning curve (Training size vs RMSE)
#reference:https://www.datacamp.com/tutorial/tutorial-learning-curves
train_sizes, train_scores, val_scores = learning_curve(
    estimator=best_model,    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 5),  # 10% to 100% of data
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

# convert negative RMSE back to positive values and average across folds
train_rmse = -train_scores.mean(axis=1)
val_rmse = -val_scores.mean(axis=1)

plt.figure(figsize=(10, 7))
plt.plot(train_sizes, train_rmse, 'o-', label="Training RMSE")
plt.plot(train_sizes, val_rmse, 'o--', label="Validation RMSE")

for x, y in zip(train_sizes, train_rmse):
    plt.text(x, y, f"{y:.1f}", fontsize=9, ha='right')

for x, y in zip(train_sizes, val_rmse):
    plt.text(x, y, f"{y:.1f}", fontsize=9, ha='left')
plt.xlabel("Number of Training Samples")
plt.ylabel("RMSE")
plt.title("Learning Curve - LightGBM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_learning_curve.png"),
        dpi=300)
plt.show()

## Top-10 feature importance bar chart
importances = best_model.feature_importances_
feature_names = X.columns
# sort features by importance (descending order)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 7))
plt.bar(range(10), importances[indices][:10], align='center')
plt.xticks(range(10), feature_names[indices][:10],
       rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Top 10 Feature Importances - LightGBM")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"),
        dpi=300)
plt.show()

## SHAP
X_for_shap = X_train if 'X_train' in globals() else X
X_shap = X_for_shap.sample(1000, random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap, check_additivity=False)
#plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, show=False)
plt.title(f"SHAP Summary Plot - {model_name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"), dpi=300)
plt.close()

# SHAP bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.title(f"SHAP Bar Plot - {model_name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_shap_bar.png"), dpi=300)
plt.close()

