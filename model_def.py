from sklearn.ensemble import GradientBoostingClassifier

# XGBClassifier alias now points to sklearn GradientBoostingClassifier
# This avoids joblib cross-version deserialization issues on Linux/Render
class XGBClassifier(GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 random_state=None, eval_metric=None, objective=None,
                 use_label_encoder=None, **kwargs):
        self.eval_metric = eval_metric
        self.objective = objective
        self.use_label_encoder = use_label_encoder
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ['eval_metric', 'objective', 'use_label_encoder']}
        )
