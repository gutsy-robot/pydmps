import pandas as pd
# from utils import get_trajectory
from dmp_discrete import DMPs_discrete
# import sys
import matplotlib.pyplot as plt
import numpy as np


def get_trajectory(csvfile):

    path_x = []
    path_y = []

    df = pd.read_csv(csvfile)
    print("read successfully..")
    recorded_path_x = df['field.x']
    recorded_path_y = df['field.y']
    for i in range(0, len(recorded_path_x)):
        if i == 0:
            path_x.append(recorded_path_x[i])
            path_y.append(recorded_path_y[i])
            print("first entry succedded..")

        else:
            if path_x[-1] != recorded_path_x[i] or path_y[-1] != recorded_path_y[i]:
                path_x.append(recorded_path_x[i])
                path_y.append(recorded_path_y[i])

    return path_x, path_y


def plot_path(x, y, n_bfs=[30, 50, 100], start=[0, 0], goal=[1, 1]):

    """

    :param x: trajectory in x
    :param y: trajectory in y
    :param n_bfs: array with number of basis functions for the DMP
    :param start: start state for the DMP
    :param goal: (x,y) goal for the DMP
    :return: plots for the

    """

    # plot given path for visualisation.

    orig_path,  = plt.plot(x, y, '--', linewidth=2)
    plot_paths = [orig_path]
    plt.plot(x[0], y[0], 'bo')
    plt.annotate("st. original", (x[0], y[0]))
    plt.plot(x[-1], y[-1], 'bo')
    plt.annotate("f original.", (x[-1], y[-1]))

    plt.plot(start[0], start[1], 'bo')
    plt.annotate("new_st", (start[0], start[1]))
    plt.plot(goal[0], goal[1], 'bo')
    plt.annotate("new_fin", (goal[0], goal[1]))

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([x, y]))
        dmp.y0[0] = start[0]
        dmp.y0[1] = start[1]
        # change the scale of the movement
        dmp.goal[0] = goal[0]
        dmp.goal[1] = goal[1]

        y_track, dy_track, ddy_track = dmp.rollout()
        # plt.subplot(211)
        plot, = plt.plot(y_track[:, 0], y_track[:, 1])
        plot_paths.append(plot)
        # plt.subplot(212)
        # plt.plot(y_track[:, 1], lw=2)
    plt.legend(plot_paths, ['original_path'] + ['%i BFs' % i for i in n_bfs], loc='lower right')

    plt.show()