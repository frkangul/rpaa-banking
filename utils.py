"""
This module contains the core functionalities for training machine learning models,
specifically focusing on logistic regression and XGBoost algorithms. It includes
functions for setting the random seed, preprocessing data, training models,
evaluating performance, and saving results. Additionally, it provides utilities for 
feature selection, hyperparameter optimization, and model interpretation through 
SHAP analysis. The module is designed to be used with a configuration object that 
dictates various aspects of the training and evaluation process.

Author: Furkan GUL
Date: 23.01.2024
"""

# IMPORT LIBS
import logging
import os
import random
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from omegaconf import DictConfig
from pandas import DataFrame, Series
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)


def seed_everything(seed: int) -> None:
    """
    Sets the seed for generating random numbers. This is used for reproducibility in experiments.

    Args:
        seed (int): The seed value to be set for generating random numbers.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_sklearn_pipeline(
    cfg: DictConfig,
    model_fnc: Callable,
    is_feats_select: bool = False,
    num_features_to_keep: int = 5,
):
    """
    Returns the inference pipeline.

    Args:
        cfg (dict): The configuration dictionary.
        model_fnc (function): The model function.
        is_feats_select (bool, optional): Whether to use feature selection. Defaults to False.
        num_features_to_keep (int, optional): The number of features to keep. Defaults to 5.

    Returns:
        tuple: The pipeline and encoder objects.
    """
    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="constant") to impute missing values
    # 2 - A OneHotEncoder() or OrdinalEncoder() step to encode the categorical feature
    config = cfg.model.config

    ordinal_cat_encoder = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("encoder", OrdinalEncoder()),
        ]
    )

    nominal_cat_encoder = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", OneHotEncoder()),
        ]
    )

    numerical_encoder = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=-999),
            ),  # -999 represents "UNKNOWN"
            ("scaler", StandardScaler()),
        ]
    )

    numerical_skewed_encoder = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=-999),
            ),  # -999 represents "UNKNOWN"
            ("scaler", PowerTransformer(method="yeo-johnson")),
        ]
    )  # There were no change in recall.

    # Let's put everything together
    encoder = ColumnTransformer(
        transformers=[
            (
                "ordinal_cat_feats",
                ordinal_cat_encoder,
                list(cfg.data.ordinal_cat_feats),
            ),
            (
                "nominal_cat_feats",
                nominal_cat_encoder,
                list(cfg.data.nominal_cat_feats),
            ),
            ("numerical_feats", numerical_encoder, list(cfg.data.numerical_feats)),
            (
                "numerical_skewed_feats",
                numerical_skewed_encoder,
                list(cfg.data.numerical_skewed_feats),
            ),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # processed_features = ordinal_cat_feats + nominal_cat_feats +
    # numerical_feats + numerical_skewed_feats

    # Create a model
    model = model_fnc(**config)

    # Use Recursive Feature Elimination (RFE) for feature selection
    # In this example, in default keep the top 5 features
    # Use SelectFromModel for feature selection
    selector = RFE(model, n_features_to_select=num_features_to_keep)

    # Create the inference pipeline. The pipeline can have many steps:
    # a step called "encoder" applying the ColumnTransformer instance that
    # we saved in the `encoder` variable, and a step called "model" with
    # the model_fnc instance that I just saved in the `model` variable etc.
    if is_feats_select is False:
        pipeline = Pipeline(steps=[("encoder", encoder), ("model", model)])
    else:
        pipeline = Pipeline(
            steps=[
                ("encoder", encoder),
                ("feature_selection", selector),
                ("model", model),
            ]
        )

    return pipeline, encoder


def save_confusion_matrix_and_precision_recall_curve(
    cfg: DictConfig, X_test: DataFrame, y_test: Series, y_pred: np.ndarray, pipe: Any
) -> None:
    """
    Saves the confusion matrix and precision-recall curve.

    Args:
        cfg (dict): The configuration dictionary.
        X_test (DataFrame): The test data.
        y_test (Series): The test labels.
        y_pred (array): The predicted labels.
        pipe (Pipeline): The pipeline object.
    """
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig(f"{cfg.model.save_result_dir}/cm.png", dpi=300)
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, name=cfg.model.name)
    plt.savefig(f"{cfg.model.save_result_dir}/pr_curve.png", dpi=300)


def log_evaluation_metrics(
    logger: logging.Logger,
    pipe: Any,
    y_test: Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> None:
    """
    Logs the evaluation metrics.

    Args:
        logger (logger): The logger object.
        pipe (Pipeline): The pipeline object.
        y_test (Series): The test labels.
        y_pred (array): The predicted labels.
        y_pred_proba (array): The predicted probabilities.
    """
    # Define classes
    classes = pipe.classes_
    class_1_label = classes[1]  # It represents 1 where there is a default.

    # Calculate metrics
    recall_class_1 = recall_score(y_test, y_pred, labels=[class_1_label], average=None)
    precision_class_1 = precision_score(
        y_test, y_pred, labels=[class_1_label], average=None
    )
    auc_pr_class_1 = average_precision_score(
        y_test, y_pred_proba
    )  # It takes probability instead of thresholded values.
    f1_class_1 = f1_score(y_test, y_pred, labels=[class_1_label], average=None)

    logger.info(f"Recall for Class 1: {round(recall_class_1[0], 3)}")
    logger.info(f"Precision for Class 1: {round(precision_class_1[0], 3)}")
    logger.info(f"AUC-PR for Class 1: {round(auc_pr_class_1, 3)}")
    logger.info(f"F1 for Class 1: {round(f1_class_1[0], 3)}")


def save_logistic_regression_curve_plot(
    cfg: DictConfig,
    clf: Any,
    X_test: DataFrame,
    y_test: Series,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Saves a plot of the logistic regression model with one feature.

    Args:
        cfg (dict): The configuration dictionary.
        clf (object): The classifier object.
        X_test (DataFrame): The test data.
        y_test (Series): The test labels.
        y_pred_proba (array): The predicted probabilities.
        y_pred (array): The predicted labels.
        feat_name (str, optional): The feature name. Defaults to "cat_number_client_calls_from_ING".
    """
    # Name of one feature to plot
    feat_name = cfg.model.logistic_regression_plot_feat_n
    # Generate a range of values from the min to the max value of the feature in the test set
    feature_range = np.linspace(
        X_test[feat_name].min(),
        X_test[feat_name].max(),
        100,
    ).reshape(-1, 1)

    # Predict probabilities for the generated range by MATHEMATICAL FORMULA
    y = feature_range * clf.coef_ + clf.intercept_  # AS SIMPLE AS THAT
    probabilities = expit(
        y
    ).ravel()  # logistic sigmoid function, is defined as expit(x) = 1/(1+exp(-x)).
    # probabilities = clf.predict_proba(feature_range)[:, 1]

    # Plot the logistic regression curve
    plt.figure(figsize=(10, 6))
    plt.plot(feature_range, probabilities, label="Logistic Regression Model")

    # Plot the threshold line
    plt.axhline(
        y=cfg.exp.threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {cfg.exp.threshold}",
    )

    # Scatter plot of the actual test set points
    plt.scatter(
        X_test[feat_name],
        y_test,
        color="black",
        label="Ground Truth",
        alpha=0.5,
    )

    # Scatter plot of the predicted probabilities for the test set
    plt.scatter(
        X_test[feat_name],
        y_pred_proba,
        color="blue",
        label="Predicted Probabilities",
        alpha=0.5,
    )

    # Scatter plot of the predicted labels for the test set
    plt.scatter(
        X_test[feat_name],
        y_pred,
        color="green",
        label="Predicted Labels",
        marker="x",
    )

    # Label the axes
    plt.xlabel("Feature Value")
    plt.ylabel("Probability of Positive Class")
    # Add a legend
    plt.legend()
    # Add title
    plt.title("Intepretation of Logistics Regression Model on Test Dataset")
    # Show the plot
    plt.savefig(f"{cfg.model.save_result_dir}/logistic_regression_model.png", dpi=300)


def save_logistic_regression_shap_analysis(
    cfg: DictConfig,
    clf: Any,
    encoder: Any,
    X_train: DataFrame,
    rfe_feats_name: List[str],
    support: np.ndarray,
) -> None:
    """
    Saves the SHAP analysis for the logistic regression model.

    Args:
        cfg (dict): The configuration dictionary.
        clf (object): The classifier object.
        encoder (object): The encoder object.
        X_train (DataFrame): The training data.
    """
    X_train_encoded = encoder.transform(X_train)
    X_train_encoded = pd.DataFrame(X_train_encoded[:, support], columns=rfe_feats_name)

    explainer = shap.LinearExplainer(clf, X_train_encoded)
    shap_values = explainer.shap_values(X_train_encoded)
    # Visualize SHAP summary plot
    plt.close("all")
    shap.summary_plot(shap_values, X_train_encoded, show=False)
    plt.savefig(
        f"{cfg.model.save_result_dir}/logistic_regression_shap_analysis.png", dpi=300
    )
    plt.close("all")
    shap.summary_plot(shap_values, X_train_encoded, plot_type="bar", show=False)
    plt.savefig(
        f"{cfg.model.save_result_dir}/logistic_regression_shap_analysis_bar.png",
        dpi=300,
    )


def save_xgboost_shap_analysis(cfg: Dict, pipe: Any, X_train: DataFrame) -> None:
    """
    Saves the SHAP analysis for the XGBoost model.

    Args:
        cfg (dict): The configuration dictionary.
        pipe (Pipeline): The pipeline object.
        X_train (DataFrame): The training data.
    """
    # Get the Booster object from XGBoost
    booster = pipe.get_booster()

    # Get contributions using Booster's predict method with pred_contribs and make SHAP analysis
    shap_values = booster.predict(
        xgb.DMatrix(X_train, enable_categorical=True), pred_contribs=True
    )[:, :-1]
    # Visualize SHAP summary plot
    plt.close("all")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"{cfg.model.save_result_dir}/xgboost_shap_analysis.png", dpi=300)
    # This takes the average of the SHAP value magnitudes across
    # the dataset and plots it as a simple bar chart.
    plt.close("all")
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig(f"{cfg.model.save_result_dir}/xgboost_shap_analysis_bar.png", dpi=300)


def save_xgboost_tree_visualization(
    cfg: DictConfig, logger: logging.Logger, pipe: Any
) -> None:
    """
    Saves a tree from the XGBoost model.

    Args:
        cfg (dict): The configuration dictionary.
        logger (logger): The logger object.
        pipe (Pipeline): The pipeline object.
    """
    # Get the number of trees (boosting rounds)
    num_trees = pipe.n_estimators

    logger.info(f"The XGBoost model has {num_trees} trees.")

    # Specify the index of the tree you want to visualize (e.g., 0 for the first tree)
    tree_index = 0

    # Set figure size and DPI for better quality
    _, ax = plt.subplots(figsize=(20, 10), dpi=300)

    # Plot the specified tree
    xgb.plot_tree(pipe, num_trees=tree_index, ax=ax)
    plt.savefig(f"{cfg.model.save_result_dir}/sample_tree.png", dpi=300)


def save_feature_target_correlation(
    cfg: DictConfig, feature_names: np.ndarray, encoder: Any, X: DataFrame, y: Series
) -> None:
    """
    Saves the correlation between features and the target variable.

    Args:
        cfg (dict): The configuration dictionary.
        feature_names (numpy.ndarray): array of feature names.
        encoder (object): The encoder object.
        X (DataFrame): The input data.
        y (Series): The target variable.
    """
    # Use the pipeline to transform the input data
    encoded_features = encoder.transform(X)

    # Convert the result to a DataFrame for better visualization
    encoded_features_df = pd.DataFrame(encoded_features, columns=feature_names)
    encoded_features_df[cfg.data.target_var] = y  # Add label to df

    # Create correlations
    correlation_matrix = encoded_features_df.corr()

    # Calculate and sort correlation with label
    target_correlations = (
        correlation_matrix[cfg.data.target_var]
        .sort_values(ascending=False)
        .reset_index()
    )
    target_correlations.to_csv(
        f"{cfg.model.save_result_dir}/target_corr.csv", index=False
    )


def log_features_selected_by_rfe(
    logger: logging.Logger, pipe: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Logs the features selected by Recursive Feature Elimination (RFE).

    Args:
        logger (logger): The logger object.
        pipe (Pipeline): The pipeline object.
    Return:
        feature_names (numpy.ndarray): array of feature names
    """
    # Find which features are selected by RFE
    support = pipe.named_steps["feature_selection"].support_

    # Extract feature names
    feature_names = pipe.named_steps["encoder"].get_feature_names_out()

    assert len(pipe.named_steps["feature_selection"].support_) == len(
        pipe.named_steps["encoder"].get_feature_names_out()
    ), "ERROR: TOTAL FEATURE NUMBER IS NOT MATCHED"
    rfe_feats_name = np.array(feature_names)[support]
    logger.info(
        f"Features selected by RFE and used by our model: {rfe_feats_name.tolist()}"
    )
    return feature_names, rfe_feats_name, support


def log_hyperparameter_results(
    logger: logging.Logger, study: optuna.study.Study
) -> None:
    """
    Logs the results of hyperparameter optimization.

    This function reports the number of completed trials, the best trial's score,
    and its hyperparameters. It is designed to work with an Optuna study object.

    Args:
        logger (logger): The logger object.
        study (optuna.study.Study): The study object containing the results of the
        hyperparameter optimization.
    """
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    logger.info(f"  Number of estimators: {trial.user_attrs['n_estimators']}")


def objective(
    trial: optuna.trial.Trial,
    cfg: DictConfig,
    X_train: DataFrame,
    y_train: Series,
    y: Series,
) -> float:
    """
    Defines the objective function for hyperparameter optimization with Optuna.

    This function is called for each trial and is responsible for training the model
    with the given hyperparameters, evaluating it using stratified K-fold cross-validation,
    and returning the validation score that Optuna will aim to optimize.

    Args:
        trial (optuna.trial.Trial): An individual trial object with methods to suggest
        hyperparameters.
        cfg (Config): A configuration object containing model configuration and other settings.
        X_train (DataFrame): The training data features.
        y_train (Series): The training data labels.
        y (Series): The complete set of labels, used for calculating class weights
        in imbalanced datasets.

    Returns:
        float: The best cross-validation score achieved by the model with
        the suggested hyperparameters.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "lambda": trial.suggest_int("lambda", 1, 10),
        "scale_pos_weight": y.value_counts()[0] / y.value_counts()[1],
        **cfg.model.config,
    }
    param.pop("n_estimators")  # It is already defined in above

    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=cfg.model.config.n_estimators,  # it means n_estimators
        nfold=cfg.model.n_folds,
        stratified=True,
        early_stopping_rounds=cfg.model.early_stop_rounds,
        seed=cfg.exp.seed,
        verbose_eval=False,
    )

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(xgb_cv_results))

    # Save cross-validation results.
    filepath = os.path.join(f"{cfg.model.save_result_dir}/{cfg.model.cv_result_dir}",\
        f"{trial.number}.csv")
    xgb_cv_results.to_csv(filepath, index=False)

    # Extract the best score.
    best_score = xgb_cv_results["test-aucpr-mean"].values[-1]
    return best_score

