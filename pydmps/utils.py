import pandas as pd
# from utils import get_trajectory
from dmp_discrete import DMPs_discrete
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


def plot_path(x, y, n_bfs=[30], start=[0, 0], goal=[1, 1]):

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

    # plt.plot(start[0], start[1], 'bo')
    # plt.annotate("new_st", (start[0], start[1]))
    # plt.plot(goal[0], goal[1], 'bo')
    # plt.annotate("new_fin", (goal[0], goal[1]))

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([x, y]))
        dmp.y0[0] = x[0]
        dmp.y0[1] = y[0]
        # change the scale of the movement
        dmp.goal[0] = x[-1]
        dmp.goal[1] = y[-1]

        y_track, dy_track, ddy_track = dmp.rollout()
        plot, = plt.plot(y_track[:, 0], y_track[:, 1])
        plot_paths.append(plot)
        plt.plot(5.6, 6.6, 'bo')
        plt.annotate("obstacle1", (5.6, 6.6))
        plt.plot(4.0, 8.0, 'bo')
        plt.annotate("obstacle2", (4.0, 8.0))
    # plt.legend(plot_paths, ['original_path'] + ['%i BFs' % i for i in n_bfs], loc='lower right')

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

        dmp.imitate_path(y_des=np.array([x, y]))
        dmp.y0[0] = x[0]
        dmp.y0[1] = y[0]
        # change the scale of the movement
        dmp.goal[0] = x[-1]
        dmp.goal[1] = y[-1]
        print(dmp.y, dmp.dy)
        y_track, dy_track, ddy_track = dmp.rollout(external_force=avoid_obstacles, obstacles=np.array([[5.6, 6.6]]))
        plot, = plt.plot(y_track[:, 0], y_track[:, 1])
        plot_paths.append(plot)

    plt.legend(plot_paths, ['original_path', 'DMP', 'DMP_obstacle'], loc='lower right')
    plt.show()


def avoid_obstacles(y, dy, goal, obstacles=np.array([[5.6, 6.6], [4.0, 8.0]]), beta=20.0 / np.pi, gamma=100):

    print("avoid obstacles called..")

    R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                         [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])

    p = np.zeros(2)
    # print("y is: ", y)
    # print("dy is: ", dy)
    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        if np.linalg.norm(dy) > 1e-5:
            # print("obstacle force will be non-zero..")
            # get the angle we're heading in
            # print("if condirion passed..")
            phi_dy = -np.arctan2(dy[1], dy[0])
            # print("phi_dy is: ", phi_dy)
            R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
                             [np.sin(phi_dy), np.cos(phi_dy)]])
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            dphi = gamma * phi * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval
    print("p is: ", p)
    return p
