import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def train_xgboost(X_train, y_train, X_val, y_val, params: dict):
    model = xgb.XGBRegressor(
        n_estimators      = params['n_estimators'],
        max_depth         = params['max_depth'],
        learning_rate     = params['learning_rate'],
        subsample         = params['subsample'],
        colsample_bytree  = params['colsample_bytree'],
        early_stopping_rounds = params['early_stopping_rounds'],
        eval_metric       = 'rmse',
        tree_method       = params['tree_method'],
        random_state      = 42
    )
    model.fit(
        X_train, y_train,
        eval_set  = [(X_val, y_val)],
        verbose   = False
    )

    # SHAP feature importance
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Save SHAP summary plot to reports/
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    return model, shap_values