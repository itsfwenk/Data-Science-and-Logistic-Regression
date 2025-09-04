#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import json
from typing import Tuple, Dict, List
import utils
import argparse

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, interations:  int=1000):
        self.learning_rate = learning_rate
        self.iterations = interations
        self.weights = None
        self.bias = None
        # self.cost_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


class OneVsAllClassifier:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.classifiers = {}
        self.classes = None
        self.feature_names = None
        self.scaler_params = None


    def standardize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """z-score normalization"""
        if fit:
            self.scaler_params = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
            self.scaler_params['std'] = np.where(self.scaler_params['std'] == 0, 1, self.scaler_params['std'])

        X_scaled = (X - self.scaler_params['mean']) / self.scaler_params['std']
        return X_scaled


    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
        self.classes = np.unique(y)
        self.feature_names = feature_names

        # print(f"Number of features: {X.shape[1]}")
        # print(f"Number of samples: {X.shape[0]}")

        X_scaled = self.standardize_features(X, fit=True)

        for class_label in self.classes:
            y_binary = (y == class_label).astype(int)
            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations
            )
            classifier.fit(X_scaled, y_binary)

            self.classifiers[class_label] = classifier

            # final_cost = classifier.cost_history[-1] if classifier.cost_history else "N/A"
            # print(f"Final cost for {class_label}: {final_cost:.6f}")


def clean_nan_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    numeric_df = df.select_dtypes(include=["number"])
    feature_cols = list(numeric_df.columns)

    X = df[feature_cols].values
    y = df['Hogwarts House'].values

    for i, col in enumerate(feature_cols):
        col_data = X[:, i]
        mask = ~np.isnan(col_data)
        if np.sum(mask) > 0:
            col_mean = np.nanmean(col_data)
            X[~mask, i] = col_mean
        else:
            X[:, i] = 0

    return X, y, feature_cols



def main():
    parser = argparse.ArgumentParser(description="Train logistic regression")
    parser.add_argument("dataset", type=str, help="dataset to train on")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return
    del df['Index']

    # print(f"Columns: {list(df.columns)}")
    # print(f"Houses: {df['Hogwarts House'].value_counts()}")

    X, y, feature_names = clean_nan_data(df)
    classifier = OneVsAllClassifier(learning_rate=0.1, max_iterations=1000)
    classifier.fit(X, y, feature_names)


if __name__ == "__main__":
    main()