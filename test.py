# # import argparse

# # parser = argparse.ArgumentParser(description='Process some integers.')

# # parser.add_argument('num1', type=int, help='The first number to add.')
# # parser.add_argument('num2', type=int, help='The second number to add.')
# # parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity.')

# # args = parser.parse_args()

# # result = args.num1 + args.num2
# # print('The sum of {} and {} is {}'.format(args.num1, args.num2, result))
# # if args.verbose:
# #     print('Verbose mode is enabled.')

# import argparse
# import os
# parser = argparse.ArgumentParser(description='Read a file and display its contents.')
# parser.add_argument('filepath', help='Path to the file.')
# args = parser.parse_args()
# if not os.path.exists(args.filepath):
#     parser.error(f"The file {args.filepath} does not exist.")
# with open(args.filepath, 'r') as file:
#     contents = file.read()
#     print(contents)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
print(df)
sns.pairplot(df, hue="species")
plt.show()
