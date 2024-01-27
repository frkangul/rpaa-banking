"""
This script is responsible for training a machine learning model to predict loan defaults. 
It includes functions for data preprocessing, model training, evaluation, and result
visualization. The script supports both logistic regression and XGBoost models, with
the ability to select features, handle categorical and numerical data, and apply various
transformations. Results such as feature importance, SHAP values, and performance
metrics are saved for analysis. The script ensures reproducibility by setting a global random seed.

Author: Furkan GUL
Date: 23.01.2024
"""
# IMPORT LIBS
import copy
import logging
import os
from typing import Any, Tuple

import hydra
import optuna
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Utility functions import
from utils import (
    get_sklearn_pipeline,
    log_evaluation_metrics,
    log_features_selected_by_rfe,
    log_hyperparameter_results,
    objective,
    save_confusion_matrix_and_precision_recall_curve,
    save_feature_target_correlation,
    save_logistic_regression_curve_plot,
    save_logistic_regression_shap_analysis,
    save_xgboost_shap_analysis,
    save_xgboost_tree_visualization,
    seed_everything,
)


def train_classification_model(
    logger: logging.Logger, cfg: DictConfig, X: DataFrame, y: Series
) -> Tuple[Any, DataFrame, DataFrame, Series, Series]:
    """
    Trains the classification model specified in the configuration. Supports logistic regression and XGBoost models.
    Performs a train-test split and fits the model on the training data.

    Args:
        logger (logger): The logger object.
        cfg (DictConfig): Configuration object containing model and experiment settings.
        X (DataFrame): The complete dataset used for training and testing the model.
        y (Series): The complete set of labels for the dataset.

    Returns:
        tuple: A tuple containing the trained classifier/pipeline, training data, test data,
               and their respective labels.
    """
    # Check for logistic regression model selection
    if cfg.model.name == "logistic_regression":
        # Perform train-test split based on the configuration settings
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.model.test_size, stratify=y, random_state=cfg.exp.seed
        )
        # Initialize and configure the logistic regression pipeline
        logistic_model = LogisticRegression
        classifier, _ = get_sklearn_pipeline(
            cfg,
            logistic_model,
            is_feats_select=True,
            num_features_to_keep=cfg.model.num_feats_rfe,
        )
    # Check for XGBoost model selection
    elif cfg.model.name == "xgboost":
        # Convert categorical feats to 'category', consider the case when a subset is specified
        if cfg.data.subset:
            X[list(cfg.data.cat_feats_subset)] = X[
                list(cfg.data.cat_feats_subset)
            ].astype("category")
        else:
            X[list(cfg.data.ordinal_cat_feats) + list(cfg.data.nominal_cat_feats)] = X[
                list(cfg.data.ordinal_cat_feats) + list(cfg.data.nominal_cat_feats)
            ].astype(
                "category"
            )  # The easiest way to pass categorical data into XGBoost
        # Perform train-test split based on the configuration settings
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.model.test_size, stratify=y, random_state=cfg.exp.seed
        )
        # Hyperparameter optimization with Optuna if enabled in configuration
        if cfg.model.hyperparam:
            # Create a result directory for Cross-val if not existed
            if not os.path.exists(f"{cfg.model.save_result_dir}/{cfg.model.cv_result_dir}"):
                os.mkdir(f"{cfg.model.save_result_dir}/{cfg.model.cv_result_dir}")
                logger.info(f"Cross-validation results will be saved in \
                    {cfg.model.save_result_dir}/{cfg.model.cv_result_dir} directory.")
            study = optuna.create_study(
                direction="maximize", study_name="XGBoost Classifier"
            )
            try:
                study.optimize(
                    lambda trial: objective(trial, cfg, X_train, y_train, y),
                    n_trials=cfg.model.optuna_trials_num,
                    timeout=600,  # Optimization timeout in seconds
                )
            except Exception as e:
                logger.error(f"Hyperparameter optimization failed: {e}")
                raise
            # Log the results of hyperparameter optimization
            log_hyperparameter_results(logger, study)

            # Combine the best hyperparameters with the existing configuration
            opt_conf = {**cfg.model.config, **study.best_params}
            # Initialize the XGBoost classifier with the optimized configuration
            classifier = xgb.XGBClassifier(
                scale_pos_weight=y.value_counts()[0] / y.value_counts()[1],
                enable_categorical=True,
                **opt_conf,
            )
        else:
            # Initialize the XGBoost classifier with the default configuration
            classifier = xgb.XGBClassifier(
                scale_pos_weight=y.value_counts()[0] / y.value_counts()[1],
                enable_categorical=True,
                **cfg.model.config,
            )
    try:
        # Fit the classifier on the training data
        classifier.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Failed to train the model: {e}")
        raise
    return classifier, X_train, X_test, y_train, y_test


def evaluate_model(
    logger: logging.Logger,
    cfg: DictConfig,
    classifier: Any,
    X_test: DataFrame,
    y_test: Series,
) -> Tuple[Series, Series]:
    """
    Evaluates the trained model on the test set. It computes the predicted probabilities and labels
    based on the specified threshold, saves the confusion matrix and precision-recall curve,
    and logs the evaluation metrics.

    Args:
        logger (logger): The logger object.
        cfg (DictConfig): Configuration object containing model and experiment settings.
        classifier (classifier): The trained classifier containing the preprocessing
        steps and the model.
        X_test (DataFrame): The test dataset.
        y_test (Series): The true labels for the test dataset.

    Returns:
        tuple: A tuple containing the predicted labels and predicted probabilities
        for the test dataset.
    """
    try:
        # Get predicted probabilities for the positive class
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]

        # Apply the threshold to get predicted labels.
        # Note: One can choose a lower threshold than 0.5 for optimizing recall.
        y_pred = (y_pred_proba >= cfg.exp.threshold).astype(int)
        # y_pred = classifier.predict(X_test)
    except Exception as e:
        logger.error(f"Failed to evaluate the model: {e}")
        raise
    # Save evaluation results and log metrics
    save_confusion_matrix_and_precision_recall_curve(
        cfg, X_test, y_test, y_pred, classifier
    )
    log_evaluation_metrics(logger, classifier, y_test, y_pred, y_pred_proba)
    return y_pred, y_pred_proba


def save_model_analysis_and_plots(
    logger: logging.Logger,
    cfg: DictConfig,
    classifier: Any,
    X: DataFrame,
    y: Series,
    X_train: DataFrame,
    X_test: DataFrame,
    y_test: Series,
    y_pred: Series,
    y_pred_proba: Series,
):
    """
    Saves various model results and analyses, including feature importance, SHAP analysis,
    and logistic regression plots. The behavior of this function changes depending on the
    model used.

    Args:
        logger (logger): The logger object.
        cfg (DictConfig): Configuration object containing model and experiment settings.
        classifier (Pipeline/classifier): The trained pipeline/classifier containing the 
        preprocessing steps and the model.
        X (DataFrame): The complete dataset used for training and testing the model.
        y (Series): The complete set of labels for the dataset.
        X_train (DataFrame): The training subset of the dataset.
        X_test (DataFrame): The test subset of the dataset.
        y_test (Series): The true labels for the test dataset.
        y_pred (ndarray): The predicted labels for the test dataset.
        y_pred_proba (ndarray): The predicted probabilities for the positive class
        for the test dataset.
    """
    # Handle logistic regression model analysis
    if cfg.model.name == "logistic_regression":
        # Log features selected by recursive feature elimination
        feature_names, rfe_feats_name, support = log_features_selected_by_rfe(
            logger, classifier
        )
        # Extract encoder and model from the pipeline
        encoder = classifier.named_steps["encoder"]
        clf = classifier.named_steps["model"]
        # Save feature-target correlation and logistic regression plots if applicable
        save_feature_target_correlation(cfg, feature_names, encoder, X, y)
        if cfg.model.num_feats_rfe == 1:
            save_logistic_regression_curve_plot(
                cfg, clf, X_test, y_test, y_pred_proba, y_pred
            )
        # Perform SHAP analysis for logistic regression
        save_logistic_regression_shap_analysis(
            cfg, clf, encoder, X_train, rfe_feats_name, support
        )
    # Handle XGBoost model analysis
    elif cfg.model.name == "xgboost":
        # Save XGBoost tree visualization
        save_xgboost_tree_visualization(cfg, logger, classifier)
        # Save feature importance: shape summarpy plot is better.
        # xgb.plot_importance(
        #     classifier, values_format="{v:.1f}", importance_type="gain"
        # )  # the average gain of splits which use the feature
        # plt.savefig("./xgboost_results/xgboost_feat_importance.png", dpi=300)
        # Perform SHAP analysis for XGBoost
        save_xgboost_shap_analysis(cfg, classifier, X_train)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    The main pipeline function that sets up the logging, configuration, seeds, logger, model,
    callbacks, and trainer. It then starts the training process and finally tests the model.
    """
    # A logger by hydra, see outputs dir.
    logger = logging.getLogger(__name__)
    # Create a result directory if not existed
    if not os.path.exists(cfg.model.save_result_dir):
        os.mkdir(cfg.model.save_result_dir)
        logger.info(f"Results will be saved in {cfg.model.save_result_dir} directory.")
    # Seed all random number generators for reproducibility
    seed_everything(cfg.exp.seed)
    # Load the dataset
    try:
        df = pd.read_csv(cfg.data.dir)
    except Exception as e:
        logger.error(f"Failed to load dataset from {cfg.data.dir}: {e}")
        raise
    # Prepare the dataset based on the model requirements
    if cfg.data.subset and cfg.model.name == "xgboost":
        X = copy.deepcopy(df)[
            cfg.data.cat_feats_subset
            + cfg.data.numerical_feats_subset
            + [cfg.data.target_var]
        ]
    else:
        X = copy.deepcopy(df)
    # Separate the target variable from the features
    y = X.pop(cfg.data.target_var)
    # Train the model and get the training and test sets
    classifier, X_train, X_test, _, y_test = train_classification_model(
        logger, cfg, X, y
    )

    # Evaluate the model and get predictions
    y_pred, y_pred_proba = evaluate_model(logger, cfg, classifier, X_test, y_test)

    # Save model analysis and plots
    save_model_analysis_and_plots(
        logger, cfg, classifier, X, y, X_train, X_test, y_test, y_pred, y_pred_proba
    )
    # Log the completion of the training and evaluation process
    logger.info("Model training and evaluation completed successfully.")


if __name__ == "__main__":
    # Execute the main function when the script is run
    main()
