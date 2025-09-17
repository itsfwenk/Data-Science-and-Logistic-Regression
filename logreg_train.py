#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from typing import Tuple, List
import utils
import argparse

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iterations:  int=1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def bgd_fit(self, X: np.ndarray, y: np.ndarray, args) -> None:
        """Train the logistic regression model using Batch Gradient Descent"""
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iterations):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)

            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

    def mbgd_fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> None:
        """Train the logistic regression model using Mini-batch Gradient Descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_cost = 0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                batch_size_actual = end_idx - start_idx

                z = X_batch @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                batch_cost = self.compute_cost(y_batch, y_pred)
                epoch_cost += batch_cost * batch_size_actual
                n_batches += 1

                dw = (1 / batch_size_actual) * X_batch.T @ (y_pred - y_batch)
                db = (1 / batch_size_actual) * np.sum(y_pred - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            avg_epoch_cost = epoch_cost / n_samples
            self.cost_history.append(avg_epoch_cost)

            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

    def sgd_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression model using Stochastic Gradient Descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_cost = 0

            for j in range(n_samples):
                X_sample = X_shuffled[j:j+1]
                y_sample = y_shuffled[j:j+1]

                z = X_sample @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                sample_cost = self.compute_cost(y_sample, y_pred)
                epoch_cost += sample_cost

                dw = X_sample.T @ (y_pred - y_sample)
                db = np.sum(y_pred - y_sample)

                self.weights -= self.learning_rate * dw.flatten()
                self.bias -= self.learning_rate * db

            avg_epoch_cost = epoch_cost / n_samples
            self.cost_history.append(avg_epoch_cost)

            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the logistic regression cost function"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

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


    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], args) -> None:
        self.classes = np.unique(y)
        self.feature_names = feature_names

        # print(f"Number of features: {X.shape[1]}")
        # print(f"Number of samples: {X.shape[0]}")

        X_scaled = self.standardize_features(X, fit=True)

        for class_label in self.classes:
            y_binary = (y == class_label).astype(int)
            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
            )
            if args.gd_type == 'SGD':
                classifier.sgd_fit(X_scaled, y_binary)
            elif args.gd_type == 'MBGD':
                classifier.mbgd_fit(X_scaled, y_binary, batch_size=args.batch_size)
            else:
                classifier.bgd_fit(X_scaled, y_binary)

            self.classifiers[class_label] = classifier
            # print(f"{class_label} weights : {classifier.weights}")

            # final_cost = classifier.cost_history[-1] if classifier.cost_history else "N/A"
            # print(f"Final cost for {class_label}: {final_cost:.6f}")


    def save_weights(self, filename: str) -> None:
        """Save trained weights to a JSON file"""
        weights_data = {
            'classes': self.classes.tolist(),
            'feature_names': self.feature_names,
            'scaler_params': {
                'mean': self.scaler_params['mean'].tolist(),
                'std': self.scaler_params['std'].tolist()
            },
            'classifiers': {}
        }

        for class_label, classifier in self.classifiers.items():
            weights_data['classifiers'][class_label] = {
                'weights': classifier.weights.tolist(),
                'bias': float(classifier.bias),
                'final_cost': classifier.cost_history[-1] if classifier.cost_history else None
            }

        with open(filename, 'w') as f:
            json.dump(weights_data, f, indent=2)

        # print(f"\nWeights saved to {filename}")

def clean_nan_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    numeric_df = df.select_dtypes(include=["number"])
    # numeric_df.dropna(inplace=True)
    # print(f"dataset size = {numeric_df.count()}")
    feature_cols = list(numeric_df.columns)

    # numeric_df = df.select_dtypes(include=["number"])
    # print(f"dataset size = {numeric_df.count()}")

    # sub_df = numeric_df[['Astronomy', 'Defense Against the Dark Arts', 'Herbology', 'Potions', 'Transfiguration', 'Charms']]
    # sub_df.dropna(inplace=True)
    # print(f"dataset size = {sub_df.count()}")
    # feature_cols = list(sub_df.columns)

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
    parser.add_argument("--gd_type", type=str, default="BGD", action='store', help="choose between Batch Gradient Descent (default: BGD), Stochastic Gradient Descent (SGD), Mini Batch Gradient Descent (MBGD)")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return
    del df['Index']

    # sub_df = df[['Astronomy', 'Defense Against the Dark Arts', 'Herbology', 'Potions', 'Transfiguration', 'Charms']]
    # print(sub_df)

    # print(f"Columns: {list(df.columns)}")
    # print(f"Houses: {df['Hogwarts House'].value_counts()}")

    X, y, feature_names = clean_nan_data(df)
    classifier = OneVsAllClassifier(learning_rate=0.1, max_iterations=1000)
    classifier.fit(X, y, feature_names, args)
    weights_filename = "weights.json"
    classifier.save_weights(weights_filename)


if __name__ == "__main__":
    main()