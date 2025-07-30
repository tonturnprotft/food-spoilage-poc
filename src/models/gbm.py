from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

def train(X, y):
    model = LGBMClassifier(n_estimators=200, class_weight={0:1,1:10},
                           random_state=42)
    model.fit(X, y)
    f1 = f1_score(y, model.predict(X))
    return model, f1