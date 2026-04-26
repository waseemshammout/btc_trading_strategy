from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb, lightgbm as lgb

def evaluate_model(y_true, y_pred, model_name):
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def train_base_model(X_train, y_train, X_valid=None, y_valid=None, model_name="XGB", iters=None):
    sc = None
    train_weights = compute_sample_weight("balanced", y_train)
    if y_valid is not None:
        val_weights = compute_sample_weight("balanced", y_valid)
    if iters is None:
        iters = 10000
        early = 50
    else:
        early = None
    if model_name == "LOG":
        sc = StandardScaler()
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        X_train_scaled = sc.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
    elif model_name == "XGB":
        model = xgb.XGBClassifier(
            n_estimators=iters, early_stopping_rounds=early, objective="multi:softmax", num_class=3
        )
        if X_valid is None:
            model.fit(X_train,y_train,
                # eval_set=[(X_valid, y_valid)],verbose=False,
                sample_weight=train_weights
                # sample_weight_eval_set=[val_weights]
                )
        else:
            model.fit(X_train,y_train,
                eval_set=[(X_valid, y_valid)],verbose=False,
                sample_weight=train_weights,
                sample_weight_eval_set=[val_weights])
            
    elif model_name == "LGB":
        model = lgb.LGBMClassifier(
            n_estimators=10000,
            random_state=1,
            objective="binary",
            class_weight="balanced",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    return model, sc
