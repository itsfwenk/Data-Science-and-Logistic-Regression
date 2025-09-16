#!/usr/bin/env python3

import utils
import argparse
import csv
import math

class ColumnStats:
    def __init__(self, name, values):
        self.name = name
        self.values = [float(v) for v in values if self.is_float(v) and not self.is_nan(v)]
        self.count = len(self.values)

    @staticmethod
    def is_nan(value):
        try:
            return float(value) != float(value)
        except:
            return False

    @staticmethod
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def mean(self):
        total = 0.0
        for x in self.values:
            total += x
        return total / self.count if self.count > 0 else 0.0

    def variance(self):
        mean_val = self.mean()
        variance_sum = 0.0
        for x in self.values:
            variance_sum += (x - mean_val) ** 2
        return variance_sum / self.count if self.count > 0 else 0.0

    def std(self):
        return math.sqrt(self.variance()) if self.count > 0 else 0.0

    def min(self):
        if not self.values:
            return 0.0
        m = self.values[0]
        for x in self.values[1:]:
            if x < m:
                m = x
        return m

    def max(self):
        if not self.values:
            return 0.0
        m = self.values[0]
        for x in self.values[1:]:
            if x > m:
                m = x
        return m

    def percentile(self, p):
        if not self.values:
            return 0.0
        nums_sorted = sorted(self.values)
        k = (self.count - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return nums_sorted[int(k)]
        return nums_sorted[f] * (c - k) + nums_sorted[c] * (k - f)

    def skewness(self):
        count_val = self.count
        mean_val = self.mean()
        std_val = self.std()

        skw_sum = 0.0
        for x in self.values:
            skw_sum += (x - mean_val) ** 3
        a = count_val / ((count_val - 1) * (count_val - 2))
        return ((a * skw_sum) / (std_val ** 3))

    def kurtosis(self):
        count_val = self.count
        mean_val = self.mean()
        std_val = self.std()

        kurt_sum = 0.0
        for x in self.values:
            kurt_sum += ((x - mean_val) / std_val) ** 4
        a = (count_val * (count_val + 1)) / ((count_val - 1) * (count_val - 2) * (count_val - 3))
        b = -((3 * ((count_val - 1) ** 2))) / ((count_val - 2) * (count_val - 3))
        return (a * (kurt_sum)) + b

    def describe(self):
        return {
            "Count": float(self.count),
            "Mean": self.mean(),
            "Std": self.std(),
            "Variance": self.variance(),
            "Min": self.min(),
            "25%": self.percentile(0.25),
            "50%": self.percentile(0.50),
            "75%": self.percentile(0.75),
            "IQR" : self.percentile(0.75) - self.percentile(0.25),
            "Max": self.max(),
            "Skewness": self.skewness(),
            "Kurtosis" : self.kurtosis()
        }


def main():
    parser = argparse.ArgumentParser(description="Describe dataset")
    parser.add_argument("dataset", type=str, help="dataset to describe")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        print("No numerical features found.")
        return

    stats_objs = []
    for col_name in numeric_df.columns:
        col_values = numeric_df[col_name].tolist()
        stats_objs.append(ColumnStats(col_name, col_values))

    column_width = 12

    truncated_names = [obj.name for obj in stats_objs]
    for i, name in enumerate(truncated_names):
        if len(name) > column_width - 5:
            truncated_names[i] = name[:column_width-5] + ".."

    header_names = [""] + truncated_names
    header_format = f"{{:>{column_width}}}" * (len(truncated_names) + 1)

    print(header_format.format(*header_names))
    print("-" * (column_width * (len(truncated_names) + 1)))

    value_format = f"{{:>{column_width}.2f}}" * len(truncated_names)
    stat_name_format = f"{{:<{column_width}}}"

    for stat_name in ["Count", "Mean", "Std", "Variance", "Min", "25%", "50%", "75%","IQR", "Max", "Skewness", "Kurtosis"]:
        row_values = [obj.describe()[stat_name] for obj in stats_objs]
        print(f"{stat_name_format.format(stat_name)}{value_format.format(*row_values)}")




if __name__ == "__main__":
    main()
