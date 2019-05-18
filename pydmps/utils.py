import pandas as pd
from dmp_discrete import DMPs_discrete
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, mapping
import random
import math

def get_trajectory(csvfile):
    # print("get trajectory called...")
    path_x = []
    path_y = []

    df = pd.read_csv(csvfile)
    print("read successfully..")
    recorded_path_x = df['field.x']
    recorded_path_y = df['field.y']
    for i in range(0, len(recorded_path_x)):
        if i == 0:
            print(recorded_path_x[i])

            # used it with * 10.0 for dmp_rrt.py might wanna change it back later.

            path_x.append((recorded_path_x[i]))
            path_y.append((recorded_path_y[i]))
            # print("first entry succedded..")

        else:
            if path_x[-1] != recorded_path_x[i] or path_y[-1] != recorded_path_y[i]:
                path_x.append((recorded_path_x[i]))
                path_y.append((recorded_path_y[i]))

    return path_x, path_y


def plot_path(x, y, n_bfs=[30], start=[5.0, 6.0], goal=[5.3, 8.0], obstacles=None):

    """

    :param x: trajectory in x
    :param y: trajectory in y
    :param n_bfs: array with number of basis functions for the DMP
    :param start: start state for the DMP
    :param goal: (x,y) goal for the DMP
    :param obstacles: ; list of shapely polygons
    :return: plots for the

    """

    print("plot_paths called for obstacles: ", obstacles)
    # plot given path for visualisation.
    legend_key = []
    orig_path,  = plt.plot(x, y, '--', linewidth=2)
    plot_paths = [orig_path]
    legend_key.append('original')
    plt.plot(x[0], y[0], 'bo')
    plt.annotate("st. original", (x[0], y[0]))
    plt.plot(x[-1], y[-1], 'bo')
    plt.annotate("f original.", (x[-1], y[-1]))

    plt.plot(start[0], start[1], 'bo')
    plt.annotate("new_st", (start[0], start[1]))
    plt.plot(goal[0], goal[1], 'bo')
    plt.annotate("new_fin", (goal[0], goal[1]))

    for ii, bfs in enumerate(n_bfs):
        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs, dt=0.01)

        dmp.imitate_path(y_des=np.array([x, y]))
        dmp.y0[0] = start[0]
        dmp.y0[1] = start[1]

        # change the scale of the movement

        # TODO: apply goal_check and goal_alteration
        dmp.goal[0] = goal[0]
        dmp.goal[1] = goal[1]

        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        # obst_y = list(np.linspace(7.7, 8.2, num=20)) + [8.2] * 10 + list(np.linspace(7.7, 8.2, num=20)) + [7.7] * 10
        # obst_x = [5.0] * 20 + list(np.linspace(5.0, 5.25, num=10)) + [5.25] * 20 + list(np.linspace(5.0, 5.25, num=10))
        # obst = []
        # for e in zip(obst_x, obst_y):
        #     obst.append(np.asarray(e))
        #
        # obst = np.array(obst)
        # print("obstacles are: ", obst)
        # for o in obst:
        #     plt.plot(o[0], o[1], 'bo')

        # plt.plot(5.6, 6.6, 'bo')
        # plt.annotate("obstacle1", (5.6, 6.6))
        # plt.plot(4.0, 8.0, 'bo')
        # plt.annotate("obstacle2", (4.0, 8.0))

        y_track_nc, dy_track_nc, ddy_track_nc, s = dmp.rollout()
        plot, = plt.plot(y_track_nc[:, 0], y_track_nc[:, 1])
        plot_paths.append(plot)
        legend_key.append('no_avoidance')
        for i in range(0, len(y_track_nc)):
            plt.plot(y_track_nc[i][0], y_track_nc[i][1], 'bo')

        path_points = []
        for point in y_track_nc:
            path_points.append(Point(tuple(point)))

        path_intersect = check_collision(path_points, obstacles)

        gammas = np.linspace(1.0, 200.0, 100)
        # print("gammas are: ", gammas)
        if path_intersect:
            print("Initial path intersects, searching for gamma...")
            for gamma in gammas:
                print("trying gamma= ", gamma)
                y_track, dy_track, ddy_track, obst_closest_pts = dmp.rollout(external_force=avoid_obstacles,
                                                                             obstacles=obstacles, gamma=gamma)
                # plot, = plt.plot(y_track[:, 0], y_track[:, 1])
                # plot_paths.append(plot)
                path_points = [Point(tuple(x)) for x in y_track]
                collision = check_collision(path_points, obstacles)
                # legend_key.append('gamma= ' + str(gamma))
                if not collision:
                    print("no collision for gamma: ", gamma)
                    print("dimension of y_track is: ", y_track.shape)
                    plot, = plt.plot(y_track[:, 0], y_track[:, 1])
                    plot_paths.append(plot)
                    legend_key.append('gamma= ' + str(gamma))
                    for i in range(0, len(y_track)):
                        plt.plot(y_track[i][0], y_track[i][1], 'bo')
                        for pts in obst_closest_pts[i]:
                            print("pts are: ", pts)
                            plt.plot(pts[0], pts[1], 'bo')
                            # plt.plot(pts[1][0], pts[1][1], 'bo')
                            plt.plot([y_track[i][0], pts[0]], [y_track[i][1], pts[1]], '--', linewidth=0.5)

                    break

                else:
                    print("collision occured for gamma= ", gamma)

        else:
            y_track, dy_track, ddy_track = dmp.rollout()

            plot, = plt.plot(y_track[:, 0], y_track[:, 1])
            plot_paths.append(plot)
            legend_key.append('with_obstacles')

    plt.legend(plot_paths, legend_key, loc='lower right')
    plt.show()


# def avoid_obstacles(y, dy, goal, obstacles, gamma, beta=20.0 / np.pi):
#
#     """
#
#     :param y: position
#     :param dy: velocity
#     :param goal: goal coordinates
#     :param obstacles: list of shapely polygons
#     :param beta: constant
#     :param gamma: constant
#     :return:
#     """
#
#     obst_closest_pts = []
#     pot = np.zeros(2)
#     for i in range(0, len(obstacles)):
#         obstacle = obstacles[i]
#
#         if np.linalg.norm(dy) > 1e-5:
#             phi_dy = -np.arctan2(dy[1], dy[0])
#             R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
#                              [np.sin(phi_dy), np.cos(phi_dy)]])
#
#             # calculate vector to object relative to body
#
#             pol_ext = LinearRing(obstacle.exterior.coords)
#             d = pol_ext.project(Point(tuple(y)))
#             p = pol_ext.interpolate(d)
#             obst_potential_pt = list(p.coords)[0]
#             # print("closest pt to the obstacle is: ", obst_potential_pt)
#             obst_closest_pts.append(obst_potential_pt)
#             # print("appended point is: ", obst_potential_pt)
#             obj_vec = obst_potential_pt - y
#             # rotate it by the direction we're going
#             obj_vec = np.dot(R_dy, obj_vec)
#             # calculate the angle of obj relative to the direction we're going
#             phi = np.arctan2(obj_vec[1], obj_vec[0])
#
#             dphi = gamma * np.exp(-beta * abs(phi))
#             R = np.dot(R_halfpi, np.outer(obst_potential_pt - y, dy))
#             pval = -np.nan_to_num(np.dot(R, dy) * dphi)
#
#             # print("pval is: ", pval)
#
#             # check to see if the distance to the obstacle is further than
#             # the distance to the target, if it is, ignore the obstacle
#             if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
#                 pval = 0
#
#             pot += pval
#     # print("p is: ", p)
#     # print("returned obst_closest_pts are: ", obst_closest_pts)
#     return pot, obst_closest_pts


def avoid_obstacles(y, dy, goal, obstacles, gamma, beta=20.0 / np.pi):

    """

    :param y: position
    :param dy: velocity
    :param goal: goal coordinates
    :param obstacles: list of shapely polygons
    :param beta: constant
    :param gamma: constant
    :return:
    """

    # print("gamma is: ", gamma)
    # print("avoid obstacles called..")

    R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                         [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
    obst_closest_pts = []
    pot = np.zeros(2)
    # print("y is: ", y)
    # print("dy is: ", dy)
    for i in range(0, len(obstacles)):
        # based on (Hoffmann, 2009)
        obstacle = obstacles[i]
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

            pol_ext = LinearRing(obstacle.exterior.coords)
            d = pol_ext.project(Point(tuple(y)))
            p = pol_ext.interpolate(d)
            obst_potential_pt = list(p.coords)[0]
            # print("closest pt to the obstacle is: ", obst_potential_pt)
            obst_closest_pts.append(obst_potential_pt)
            # print("appended point is: ", obst_potential_pt)
            obj_vec = obst_potential_pt - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])

            dphi = gamma * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer(obst_potential_pt - y, dy))
            pval = -np.nan_to_num(np.dot(R, dy) * dphi)

            # print("pval is: ", pval)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            pot += pval
    # print("p is: ", p)
    # print("returned obst_closest_pts are: ", obst_closest_pts)
    return pot, obst_closest_pts


def check_collision(path_points, obstacles):

    """

    :param path_points: list of shapely points
    :param obstacles: list of shapely polygons
    :return: boolean collision
    """
    collision = False

    for obs in obstacles:
        x, y = obs.exterior.coords.xy
        print("checking for obstacle: ", mapping(obs)['coordinates'])
        for point in path_points:
            if obs.contains(point):
                collision = True
                print("pt: " + str(point) + " lies inside the obstacle")
                break

            else:
                continue

            break

    return collision


def sample_dmp_normal(x_dmp, y_dmp, t_dmp):
    mean = [x_dmp, y_dmp, t_dmp]
    cov = [[20, 0, 0], [0, 20, 0], [0, 0, 0.1]]
    x, y, t = np.random.multivariate_normal(mean, cov, 1).T

    return x, y, t


def sample_uniform(minx, miny, mint, maxx, maxy, maxt):
    x = (random.random() - minx) * (maxx - minx)
    y = (random.random() - miny) * (maxy - miny)
    t = (random.random() - mint) * (maxt - mint)
    # if math.sqrt((x - 10.0) ** 2 + (y - 10.0) ** 2 + t **2) < 9.0:
    #     print("point sampled should be in the roadmap of t[0]")

    return x, y, t


def ind_max(x):
    m = max(x)
    return x.index(m)
