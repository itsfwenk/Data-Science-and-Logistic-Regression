#!/usr/bin/env python3
import utils
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Describe dataset")
    parser.add_argument("dataset", type=str, help="dataset to describe")
    args = parser.parse_args()

    df = utils.load_csv(args.dataset)
    if df is None or df.empty:
        print("Empty or invalid dataset.")
        return
    del df['Index']


    sns.set_theme(style="ticks", font_scale=0.6)
    g = sns.pairplot(df, hue="Hogwarts House",
                     height=1,
                     aspect=1,
                    #  diag_kind='hist',
                     plot_kws={'s': 10, 'alpha': 0.6})


    max_length = 8
    for ax in g.axes.flatten():
        if ax:
            ax.set_xticks([])
            ax.set_yticks([])
            ylabel = ax.get_ylabel()
            if len(ylabel) > max_length:
                ax.set_ylabel(ylabel[:max_length] + '..')

            xlabel = ax.get_xlabel()
            if len(xlabel) > max_length:
                ax.set_xlabel(xlabel[:max_length] + '..')

    plt.show()

if __name__ == "__main__":
    main()
