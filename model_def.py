from sklearn.ensemble import GradientBoostingClassifier

# Centralized custom model class to prevent joblib serialization/unpickling module namespace issues.
# Mapped to scikit-learn GradientBoostingClassifier if OpenMP is not installed.
try:
    from xgboost import XGBClassifier as OriginalXGBClassifier
    class XGBClassifier(OriginalXGBClassifier):
        pass
except Exception:
    class XGBClassifier(GradientBoostingClassifier):
        def __init__(self, n_estimators=100, learning_rate=0.1, random_state=None, eval_metric=None, use_label_encoder=None, **kwargs):
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.random_state = random_state
            self.eval_metric = eval_metric
            self.use_label_encoder = use_label_encoder
            super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **kwargs)
