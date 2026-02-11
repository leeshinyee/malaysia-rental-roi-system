#RF + XGBoost(Soft Ensemble / Average）
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import learning_curve
import shap

W = 0.6  # xgboost weight

# wrap 2 models to conduct learning curve analysis and cv
class AveragingEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model1, model2, w=W):
        self.model1 = model1 #xgboost
        self.model2 = model2 #rf
        self.w = w

    def fit(self, X, y):
        self.m1_ = clone(self.model1)
        self.m2_ = clone(self.model2)
        self.m1_.fit(X, y)
        self.m2_.fit(X, y)
        return self

    def predict(self, X):
        p1 = self.m1_.predict(X)
        p2 = self.m2_.predict(X)
        # weighted average
        return self.w * p1 + (1 - self.w) * p2


# fitted-model wrapper for deployment
class FittedAveragingEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model1_fitted, model2_fitted, w=W):
        self.model1_fitted = model1_fitted  # fitted xgboost
        self.model2_fitted = model2_fitted  # fitted rf
        self.w = w

    def fit(self, X, y):
        return self

    def predict(self, X):
        p1 = self.model1_fitted.predict(X)
        p2 = self.model2_fitted.predict(X)
        return self.w * p1 + (1 - self.w) * p2


# load data
df = pd.read_csv(r"final_dataset.csv")

drop_columns = [
    'ads_id','prop_name','completion_year','monthly_rent',
    'facilities','additional_facilities','property_type_grouped',
]

X = df.drop(columns=drop_columns)
y = df['monthly_rent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# load models (these are already tuned + fitted)
xgb_model = joblib.load("best_xgboost_model.pkl")
rf_model  = joblib.load("best_random_forest_model.pkl")

# define model
hybrid_model = FittedAveragingEnsembleRegressor(
    model1_fitted=xgb_model,
    model2_fitted=rf_model,
    w=W)

y_pred = hybrid_model.predict(X_test)

# model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

print("\nHybrid XGBoost and Random Forest")
print(f"  MAE  : {mae:.2f}")
print(f"  MSE  : {mse:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}")
print(f"  MAPE : {mape:.2f}%")

# save model
joblib.dump(hybrid_model, "best_hybrid_xgb_rf_model.pkl")
print("\nModel Saved")

model_name = "Hybrid_XGBoost_RandomForest"
output_dir = os.path.dirname(os.path.abspath(__file__))

## Actual vs Predicted scatter plot
plt.figure(figsize=(10, 7))
# x = actual rent, y = model predicted rent
plt.scatter(y_test, y_pred, alpha=0.4, s=20, color="#1f77b4")
# adjust x range
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value], [min_value, max_value],
         'r--', linewidth=2, label="Perfect Prediction")

plt.xlabel("Actual Monthly Rent", fontsize=12)
plt.ylabel("Predicted Monthly Rent", fontsize=12)
plt.title(f"Actual vs Predicted Monthly Rent - {model_name} ", fontsize=14)
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
print("graph 1")

## Residual scatter plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 7))
x_jitter = y_pred + np.random.normal(0, 25, size=len(y_pred))

plt.scatter(x_jitter, residuals,
            alpha=0.15, s=12, color="#1f77b4")

plt.axhline(0, color='red', linestyle='--', linewidth=1)

plt.xlabel("Predicted Monthly Rent")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title(f"Residual Plot - {model_name}")

# limit y range for better visualization
plt.ylim(-800, 800)

plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_residual_plot.png"),dpi=300)
plt.show()
print("graph 2")

## Learning curve (Training size vs RMSE)
xgb = clone(xgb_model)
rf  = clone(rf_model)

hybrid_estimator = AveragingEnsembleRegressor(
    model1=xgb,
    model2=rf,
    w=W
)

train_sizes, train_scores, val_scores = learning_curve(
    estimator=hybrid_estimator,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    shuffle=True,
    random_state=42
)
# convert negative RMSE to positive
train_rmse = -train_scores.mean(axis=1)
val_rmse   = -val_scores.mean(axis=1)

plt.figure(figsize=(10, 7))
plt.plot(train_sizes, train_rmse, "o-", label="Training RMSE")
plt.plot(train_sizes, val_rmse, "o--", label="Validation RMSE")

for x, yv in zip(train_sizes, train_rmse):
    plt.text(x, yv, f"{yv:.1f}", fontsize=9, ha="right")
for x, yv in zip(train_sizes, val_rmse):
    plt.text(x, yv, f"{yv:.1f}", fontsize=9, ha="left")

plt.xlabel("Number of Training Samples")
plt.ylabel("RMSE")
plt.title(f"Learning Curve - {model_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_learning_curve.png"), dpi=300)
plt.show()
print("graph 3")

X_shap = X_train.sample(1000, random_state=42)

# base model SHAP
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_xgb = explainer_xgb.shap_values(X_shap, check_additivity=False)

explainer_rf = shap.TreeExplainer(rf_model)
shap_rf = explainer_rf.shap_values(X_shap, check_additivity=False)

# weighted SHAP for hybrid
shap_hybrid = W * shap_xgb + (1 - W) * shap_rf

# Graph 4: SHAP Summary (beeswarm)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_hybrid, X_shap, show=False)
plt.title(f"SHAP Summary Plot (Weighted) - {model_name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary_weighted.png"), dpi=300)
plt.close()
print("graph 4")

## SHAP Bar (global)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_hybrid, X_shap, plot_type="bar", show=False)
plt.title(f"SHAP Bar Plot (Weighted) - {model_name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_shap_bar_weighted.png"), dpi=300)
plt.close()
print("graph 5")

## Top-10 from weighted SHAP (mean |SHAP|)
mean_abs_shap = np.abs(shap_hybrid).mean(axis=0)
indices = np.argsort(mean_abs_shap)[::-1]

plt.figure(figsize=(10, 7))
plt.bar(range(10), mean_abs_shap[indices][:10], align='center')
plt.xticks(range(10), X_shap.columns[indices][:10], rotation=45, ha='right')
plt.ylabel("Mean |SHAP value|")
plt.title(f"Top 10 Features (Weighted SHAP) - {model_name}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{model_name}_top10_weighted_shap.png"), dpi=300)
plt.close()
print("graph 6")