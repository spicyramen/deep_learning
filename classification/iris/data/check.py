import pandas as pd
import matplotlib.pyplot as plt

FILENAME = '../data/iris.csv'


def read_file(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except IOError as e:
        print e


def visualize(iris):
    color_map = dict(zip(iris.species.unique(), ['red', 'blue', 'green']))
    for species, group in iris.groupby('species'):
        plt.scatter(group['petalLength'], group['sepalLength'],
                    color=color_map[species],
                    alpha=0.3,
                    edgecolor=None,
                    label=species)
    plt.legend(frameon=True, title='species')
    plt.xlabel('petalLength')
    plt.ylabel('sepalLength')
