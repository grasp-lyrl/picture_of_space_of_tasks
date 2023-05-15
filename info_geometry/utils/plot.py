import os
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from tabulate import tabulate


def plt_style():
    plt.style.use('seaborn-v0_8-whitegrid')

    sns.set(context='poster',
            style='ticks',
            font_scale=0.65,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})


def print_lengths(len_list, task, task_header='ImageNet task'):

    header = [task_header, "Reimann Length", "Standard error"]
    pm = u'\u00B1'

    table = []
    for idx, ln in enumerate(len_list):
        riemann = "%0.3f" % ln[0]
        std = "%c %0.3f" % (pm, ln[1])
        table.append([task[idx], riemann, std])

    print(tabulate(table, headers=header))


def plot_trajectory_distances(traj_info, names, fname):
    plt_style()

    for info in traj_info:
        plt.plot(info[0], info[1])

    plt.title("Distance between Trajectories")
    plt.xlim([0, 1])
    plt.xlabel("Progress")
    plt.ylabel("Bhattacharrya distance")
    plt.legend(names)

    fdir = '/'.join(fname.split("/")[:-1])
    os.makedirs(fdir, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")


def plot_inpca():
    pass
