import numpy as np, lightgbm as lgb
from sklearn.metrics import f1_score
def train_fusion(X_train, image_prob_train, y_train):
    X_aug = np.column_stack([X_train, image_prob_train])
    model = lgb.LGBMClassifier(n_estimators=200, random_state=42).fit(X_aug, y_train)
    return model
def test_fusion(model, X_test, image_prob_test, y_test):
    X_aug = np.column_stack([X_test, image_prob_test])
    return f1_score(y_test, model.predict(X_aug))

