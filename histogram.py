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

    ravenclaw = df.loc[df['Hogwarts House'] == 'Ravenclaw']
    slytherin = df.loc[df['Hogwarts House'] == 'Slytherin']
    gryffindor = df.loc[df['Hogwarts House'] == 'Gryffindor']
    hufflepuff = df.loc[df['Hogwarts House'] == 'Hufflepuff']

    numeric_r = ravenclaw.select_dtypes(include=["number"])
    numeric_s = slytherin.select_dtypes(include=["number"])
    numeric_g = gryffindor.select_dtypes(include=["number"])
    numeric_h = hufflepuff.select_dtypes(include=["number"])


    raven_stats = []
    for col_name in numeric_r.columns:
        col_values = numeric_r[col_name].tolist()
        raven_stats.append(ColumnStats(col_name, col_values))

    slyth_stats = []
    for col_name in numeric_s.columns:
        col_values = numeric_s[col_name].tolist()
        slyth_stats.append(ColumnStats(col_name, col_values))

    gryff_stats = []
    for col_name in numeric_g.columns:
        col_values = numeric_g[col_name].tolist()
        gryff_stats.append(ColumnStats(col_name, col_values))

    huffle_stats = []
    for col_name in numeric_h.columns:
        col_values = numeric_h[col_name].tolist()
        huffle_stats.append(ColumnStats(col_name, col_values))

    raven_std = {}
    for obj in raven_stats:
        raven_std[obj.name] = obj.describe()["Std"]

    slyth_std = {}
    for obj in slyth_stats:
        slyth_std[obj.name] = obj.describe()["Std"]

    gryff_std = {}
    for obj in gryff_stats:
        gryff_std[obj.name] = obj.describe()["Std"]

    huffle_std = {}
    for obj in huffle_stats:
        huffle_std[obj.name] = obj.describe()["Std"]

    sum_std = {}
    for name, r, s, g, h in zip(raven_std, raven_std.values(), slyth_std.values(), gryff_std.values(), huffle_std.values()):
        sum_std[name] = r+s+g+h
    # print(sum_std)
    most_homogeneous_course = min(sum_std, key=sum_std.get)
    # print(most_homogeneous_course)
    homogeneous_course_data = df[['Hogwarts House', most_homogeneous_course]]
    print(homogeneous_course_data)

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    for house in homogeneous_course_data['Hogwarts House'].unique():
        subset = homogeneous_course_data[homogeneous_course_data['Hogwarts House'] == house]
        sns.histplot(
            data=subset,
            x=most_homogeneous_course,
            kde=True,
            label=house,
            alpha=0.6
        )

    plt.title(f'Score Distribution for {most_homogeneous_course} by Hogwarts House', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Hogwarts House')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    plt.show()

    print(f"\nAnalysis Summary:")
    print(f"Based on the standard deviation of scores, the course with the most homogeneous distribution between all four houses is: {most_homogeneous_course}")
    print("\nThis means the scores in this course are most similarly spread out across all houses.")

if __name__ == "__main__":
    main()


# grouped_by_house = df.groupby('Hogwarts House')
# house_std = grouped_by_house[course_columns].std()
# course_std_sum = house_std.sum()
# most_homogeneous_course = course_std_sum.idxmin()
# homogeneous_course_data = df[['Hogwarts House', most_homogeneous_course]]