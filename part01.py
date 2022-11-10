#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: Jakub Krivanek - xkriva30

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny
predstavene na prednasce
"""


from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def integrate(x: np.array, y: np.array) -> float:
    """
    Calculate integral and return the result as float
    """
    return np.sum((x[1:] - x[:-1]) * (y[:-1] + y[1:]) / 2)


def generate_graph(a: List[float], show_figure: bool = False,
                   save_path: str | None = None):
    """Generate graph for function f_a(x) = a * x^2 defined in interval <-3,3>

    Keyword arguments:
        a -- values for function
        show_figure -- display graph (default False)
        save_path -- save graph (default None)
    """
    x = np.linspace(-3, 3, 100)
    x_2 = np.power(x, 2).reshape(-1, 1)
    a = np.full([x_2.shape[0], len(a)], a)
    f = a * x_2

    y_1 = f[:, 0]
    y_2 = f[:, 1]
    y_m2 = f[:, 2]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel(r'$f_a(x)$')
    ax.plot(x, y_1, color='tab:blue', label=r'$y_{1.0}(x)$')
    ax.plot(x, y_2, color='tab:orange', label=r'$y_{2.0}(x)$')
    ax.plot(x, y_m2, color='tab:green', label=r'$y_{-2.0}(x)$')
    ax.annotate(
        r'$\int f_{2.0}(x)dx$', xy=(3, y_2[-1]), verticalalignment='center')
    ax.annotate(
        r'$\int f_{1.0}(x)dx$', xy=(3, y_1[-1]), verticalalignment='center')
    ax.annotate(
        r'$\int f_{-2.0}(x)dx$', xy=(3, y_m2[-1]), verticalalignment='center')
    ax.set_ylim(-20, 20)
    ax.spines.bottom.set_position(('data', -20))
    ax.set_xlim(-3, 4)
    ax.xaxis.get_major_ticks()[-1].set_visible(False)
    ax.spines['right'].set_position(('data', 4))
    ax.fill_between(x, y_2, 0, color='tab:orange', alpha=0.1, linewidth=0)
    ax.fill_between(x, y_m2, 0, color='tab:green', alpha=0.1, linewidth=0)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if show_figure:
        plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """Generate 3 graphs for functions
    f_1(t) = 0.5 * sin(pi * t / 50)
    f_1(t) = 0.25 * sin(pi * t)
    f_3(t) = f1(t) + f2(t)

    Keyword arguments:
        show_figure -- display graph (default False)
        save_path -- save graph (default None)
    """
    t = np.linspace(0, 100, 10000)
    f1 = 0.5 * np.sin(np.pi / 50 * t)
    f2 = 0.25 * np.sin(np.pi * t)
    f3 = f1 + f2

    fig = plt.figure(constrained_layout=True, figsize=(10, 18))
    axes = (fig.add_gridspec(nrows=3).subplots())
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_xlabel('t')
        ax.set_ylim(-0.8, 0.8)
        ax.spines.bottom.set_position(('data', -0.8))
        ax.set_yticks(np.arange(-0.8, 1.2, 0.4))
        ax.set_xlim(0, 100)
    ax1.set_ylabel(r'$f_1(t)$')
    ax2.set_ylabel(r'$f_2(t)$')
    ax3.set_ylabel(r'$f_1(t) + f_2(t)$')
    ax1.plot(t, f1)
    ax2.plot(t, f2)
    ax3.plot(t, np.ma.masked_less(f3, f1), 'g')
    ax3.plot(t, np.ma.masked_greater_equal(f3, f1), 'r')

    if show_figure:
        plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def download_data(url="https://ehw.fit.vutbr.cz/izv/temp.html"):
    """Download data from url - expecting xml
    Return data in list of tuples - each tuple is a month

    Keyword arguments:
        url -- url to data (default https://ehw.fit.vutbr.cz/izv/temp.html)
    """
    r = requests.get(url=url)
    soup = BeautifulSoup(r.content, features="lxml")
    data = list()
    tr = soup.find_all('tr', class_='ro1')
    for row in tr:
        p = row.find_all('p')
        temp = np.empty(len(p) - 2, dtype=float)
        row_tuple = {}
        for i, j in enumerate(p):
            text = j.text
            if i == 0:
                row_tuple["year"] = int(text)
            elif i == 1:
                row_tuple["month"] = int(text)
            else:
                temp[i - 2] = float(text.replace(',', '.'))
        row_tuple["temp"] = temp
        data.append(row_tuple)
    return data


def get_avg_temp(data, year=None, month=None) -> float:
    """Return average temperature from data based on selection

    Keyword arguments:
        data -- data containing temperatures
        year -- selected year (default None)
        month -- selected month (default None)
    """
    if (year):
        data = [x["temp"] for x in data if x["year"] == year]
    elif (month):
        data = [x["temp"] for x in data if x["month"] == month]
    elif (year and month):
        data = [x["temp"] for x in data
                if x["year"] == year and x["month"] == month]
    else:
        data = [x["temp"] for x in data]
    data = np.concatenate(data, axis=0)
    return (np.sum(data) / data.size)
