import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import LabelEncoder


def train_predict():
    train_df = pd.read_csv(r".\data\train.csv")
    test_df = pd.read_csv(r".\data\test.csv")

    for c in train_df.columns:
        if train_df[c].dtype == "object":
            lbl = LabelEncoder()
            lbl.fit(list(train_df[c].values) + list(test_df[c].values))
            train_df[c] = lbl.transform(list(train_df[c].values))
            test_df[c] = lbl.transform(list(test_df[c].values))

    X_train = train_df.drop(["ID", "y"], axis=1)
    y_train = train_df.y
    X_test = test_df.drop(["ID"], axis=1)

    model_elastic = ElasticNetCV()
    model_elastic.fit(X_train, y_train)
    preds = model_elastic.predict(X_test)
    submit(test_df, preds)


def submit(test_df, preds):
    submission_path = r".\data\sample_submission.csv"
    submission = pd.DataFrame()
    submission["ID"] = test_df.ID
    submission["y"] = preds
    submission.to_csv(submission_path, index=False)


if __name__ == "__main__":
    train_predict()
