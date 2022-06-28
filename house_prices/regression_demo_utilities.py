"""" Data preprocessing, model training, evaluation and hyperparam optimization"""

import pandas as pd
import numpy as np
import datetime
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
def eval_model(y_train_true, y_train_pred, y_test_true, y_test_pred, metric, **kwargs):
    """Calculate metrics for train and test sets"""
    try:
        return {
            f"{metric.__name__}_train": metric(y_train_true, y_train_pred, **kwargs),
            f"{metric.__name__}_test": metric(y_test_true, y_test_pred, **kwargs),
        }
    except:
        import pdb

        pdb.set_trace()


def create_predictions(model, X_train, X_test, per_m2: bool = False):
    """Create predictions for train and test"""
    if per_m2:
        res = (
            model.predict(X_train) * X_train["GrLivArea"],
            model.predict(X_test) * X_test["GrLivArea"],
        )

    else:
        res = (model.predict(X_train), model.predict(X_test))
    try:
        assert np.isnan(res[0]).sum() == 0

    except:
        import pdb

        pdb.set_trace()
    return res


def setup_regression(train, train_features, core_model, per_m2=False, log_y=False):
    """Create a model and separate the data into X, y train and test"""

    if per_m2:
        y_name = "SalePrice_per_GrLivArea"
    else:
        y_name = "SalePrice"

    X_train, X_test, y_train, y_test = train_test_split(
        train[train_features], train[y_name], test_size=0.33, random_state=42
    )
    X_train = pd.DataFrame(X_train, columns=train_features)
    X_test = pd.DataFrame(X_test, columns=train_features)

    if log_y:
        model = TransformedTargetRegressor(
            regressor=core_model, func=np.log, inverse_func=np.exp
        )
    else:
        model = core_model
    numerical_features=list(X_train.select_dtypes(np.number).columns)
    categorical_features = [col for col in train_features if col not in numerical_features]
    model_pipeline = Pipeline(
        [
            #("imputer", SimpleImputer(strategy="most_frequent")),
            ('num-cat-split',ColumnTransformer(
                [
                    ("numerical", StandardScaler(), numerical_features),
                    ("categorical", OneHotEncoder(), categorical_features),
                ],
                verbose_feature_names_out=False,
            )),
            ("model", model),
        ]
    )
    model_pipeline.fit(X_train, y_train)
    if per_m2:
        return (
            model_pipeline,
            X_train,
            X_test,
            y_train * X_train["GrLivArea"],
            y_test * X_test["GrLivArea"],
        )
    else:
        return model_pipeline, X_train, X_test, y_train, y_test


def calculate_metrics_for_pipeline(train, train_features, core_model, per_m2, log_y):
    model_pipeline, X_train, X_test, y_train, y_test = setup_regression(
        train,
        train_features,
        core_model=core_model,
        per_m2=per_m2,
        log_y=log_y,
    )

    y_train_pred, y_test_pred = create_predictions(
        model_pipeline, X_train, X_test, per_m2=per_m2
    )

    return {
        **eval_model(y_train, y_train_pred, y_test, y_test_pred, r2_score),
        **eval_model(
            y_train,
            y_train_pred,
            y_test,
            y_test_pred,
            mean_squared_error,
            squared=False,
        ),
    }
