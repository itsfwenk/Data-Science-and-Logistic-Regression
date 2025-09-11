#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from typing import Tuple, List
import utils
import argparse


class LogRegPredictor():
    def __init__(self, to_predict: np.ndarray, logreg_params: dict):
        self.to_predict = to_predict
        self.logreg_params = logreg_params

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def standardize_features(self, X: np.ndarray) -> np.ndarray:
        """z-score normalization"""
        mean = self.logreg_params['scaler_params']['mean']
        std = self.logreg_params['scaler_params']['std']
        X_scaled = (X - mean) / std
        return X_scaled

    def predict(self) -> pd.DataFrame:
        predictions = {
        "Hogwarts House": []
        }
        proba = {
            "Hufflepuff": [],
            "Ravenclaw": [],
            "Gryffindor": [],
            "Slytherin": []
        }
        X_scaled  = self.standardize_features(self.to_predict)

        for student in X_scaled :
            best_pred = 0
            for house, params in self.logreg_params['classifiers'].items():
                weights = params['weights']
                bias = params['bias']
                z = student @ weights + bias
                y_pred = self.sigmoid(z)
                if (best_pred < y_pred):
                    best_pred = y_pred
                    belong_to = house
                proba[house].append(y_pred)
            predictions["Hogwarts House"].append(belong_to)
        predict_df = pd.DataFrame(predictions)
        proba_df = pd.DataFrame(proba)
        proba_df.to_csv('proba.csv', index_label='Index')
        return predict_df
        # print(predict_df)









def clean_nan_data(df: pd.DataFrame) -> np.ndarray:
    numeric_df = df.select_dtypes(include=["number"])
    feature_cols = list(numeric_df.columns)

    X = df[feature_cols].values

    for i, col in enumerate(feature_cols):
        col_data = X[:, i]
        mask = ~np.isnan(col_data)
        if np.sum(mask) > 0:
            col_mean = np.nanmean(col_data)
            X[~mask, i] = col_mean
        else:
            X[:, i] = 0

    return X

def main():
    parser = argparse.ArgumentParser(description="Predict Hogwart Houses from a Dataset")
    parser.add_argument("dataset", type=str, help="dataset to predict")
    parser.add_argument("weights", type=str, help="predicted logistic regressions parameters")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return
    df.drop(['Index', 'Hogwarts House'], axis=1, inplace=True)

    to_predict = clean_nan_data(df)

    with open(args.weights, "r") as f:
        model = json.load(f)

    logreg_params = {
            'classes': model['classes'],
            'feature_names': model['feature_names'],
            'scaler_params': {
                'mean': model['scaler_params']['mean'],
                'std': model['scaler_params']['std']
            },
            'classifiers': model['classifiers']
        }

    # print(logreg_params)

    predictor = LogRegPredictor(to_predict, logreg_params)
    # print(predictor.logreg_params.classes)
    predictions = predictor.predict()
    predictions.to_csv('houses.csv', index_label='Index')





if __name__ == "__main__":
    main()