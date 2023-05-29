import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from tabulate import tabulate
from matplotlib.colors import to_rgb, to_hex


def plt_style():
    plt.style.use('seaborn-v0_8-whitegrid')

    sns.set(context='poster',
            style='ticks',
            font_scale=0.65,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})


def cdark(rgb):
    rgb = to_rgb(rgb)
    rgb = 0.4 + 0.5 * np.array(rgb) 
    return to_hex(rgb)


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


def plot_inpca(tab, name, cols, fname):
    """
    Plot as InPCA
    """
    all_plots = []
    num_models = len(name.keys())

    for i in range(num_models):

        mt = tab[tab['Model'] == name[i]]
        plot = go.Scatter3d(
            x=mt['PC 1'], y=mt['PC 2'], z=mt['PC 3'],
            name=name[i],
            mode='markers',
            marker=dict(size=3, color=cols[i], opacity=0.7),
            line=dict(color='black', width=10),
            showlegend=True
        )
        all_plots.append(plot)

        mt_avg = mt.groupby('Epoch').mean(numeric_only=True)
        plot = go.Scatter3d(
            x=mt_avg['PC 1'], y=mt_avg['PC 2'], z=mt_avg['PC 3'],
            marker=dict(size=0.01, color=cdark(cols[i]), opacity=0.4),
            line=dict(color=cdark(cols[i]), width=7),
            showlegend=False
        )
        all_plots.append(plot)

    fig = go.Figure(data=all_plots)
    fig.update_layout(
        template='plotly_white',
        scene = dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10), 
        scene_camera=dict(
            eye=dict(x=1.65, y=1.2, z=1.9)),
        legend= {'itemsizing': 'constant'},
        scene_aspectmode='cube'
    )

    fdir = '/'.join(fname.split("/")[:-1])
    os.makedirs(fdir, exist_ok=True)
    fig.write_html(fname)
