#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Jakub Krivanek - xkriva30

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    return np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2)


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None=None):
    step = 0.05
    x = np.arange(-3, 3 + step, step)
    x_2 = np.power(x, 2).reshape(-1, 1)
    f = a * x_2
    y_1 = f[:, 0]
    y_2 = f[:, 1]
    y_m2 = f[:, 2]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel(r'$f_a(x)$')
    ax.plot(x, y_1, color='tab:blue', label=r'$y_{1.0}(x)$')
    ax.plot(x, y_2, color='tab:orange', label=r'$y_{2.0}(x)$')
    ax.plot(x, y_m2, color='tab:green', label=r'$y_{-2.0}(x)$')
    ax.annotate(r'$\int f_{2.0}(x)dx$', xy=(3, y_2[-1]), verticalalignment='center')
    ax.annotate(r'$\int f_{1.0}(x)dx$', xy=(3, y_1[-1]), verticalalignment='center')
    ax.annotate(r'$\int f_{-2.0}(x)dx$', xy=(3, y_m2[-1]), verticalalignment='center')
    ax.set_ylim(-20, 20)
    ax.set_xlim(-3, 4)
    ax.spines.bottom.set_position(('data', -20))
    ax.xaxis.get_major_ticks()[-1].set_visible(False)
    ax.spines['right'].set_position(('data', 4))
    ax.fill_between(x, y_2, 0, color = 'tab:orange', alpha=0.1, linewidth=0)
    ax.fill_between(x, y_m2, 0, color = 'tab:green', alpha=0.1, linewidth=0)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    if show_figure:
        plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight") 



def generate_sinus(show_figure: bool=False, save_path: str | None=None):
    pass


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    pass


def get_avg_temp(data, year=None, month=None) -> float:
    pass

generate_graph([1.0, 2.0, -2.0], True, False)
# generate_graph([1.0, 2.0, -2.0], True, "graph.png")
