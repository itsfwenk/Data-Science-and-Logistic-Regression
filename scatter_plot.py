#!/usr/bin/env python3
import utils
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from describe import ColumnStats

def main():
    parser = argparse.ArgumentParser(description="Describe dataset")
    parser.add_argument("dataset", type=str, help="dataset to describe")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return
    del df['Index']

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        print("No numerical features found.")
        return

    matrix  = numeric_df.corr().abs()
    # plt.figure(figsize=(8,6))
    # sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    # plt.title("Correlation Heatmap")
    # plt.show()

    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    top_correlation_series = get_top_abs_correlations(numeric_df, 1)
    most_correlated_pair = top_correlation_series.index[0]

    # df_to_plot = pd.DataFrame({
    #     most_correlated_pair[0]: numeric_df[most_correlated_pair[0]],
    #     most_correlated_pair[1]: numeric_df[most_correlated_pair[1]],
    # })
    # sns.scatterplot(data=df_to_plot, x=most_correlated_pair[0], y=most_correlated_pair[1])
    # plt.title(f"Correlation between {most_correlated_pair[0]} and {most_correlated_pair[1]}")
    # plt.xlabel(most_correlated_pair[0])
    # plt.ylabel(most_correlated_pair[1])
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 6))

    sns.regplot(
        x=numeric_df[most_correlated_pair[0]],
        y=numeric_df[most_correlated_pair[1]],
        scatter_kws={'alpha': 0.7, 's': 100},
        line_kws={'color': 'red', 'lw': 2}
    )
    plt.title(
        f'Scatter Plot of the Two Most Correlated Features\n({most_correlated_pair[0]} vs. {most_correlated_pair[1]})',
        fontsize=16
    )
    plt.xlabel(most_correlated_pair[0], fontsize=12)
    plt.ylabel(most_correlated_pair[1], fontsize=12)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()