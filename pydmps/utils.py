import pandas as pd
from dmp_discrete import DMPs_discrete
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, mapping
import random
import math
from math import sqrt

def get_trajectory(csvfile):
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

    R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                         [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
    obst_closest_pts = []
    pot = np.zeros(2)

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


def sample_dmp_normal(x_dmp, y_dmp, t_dmp, variance=0.04, time_variance=0.04):
    mean = [x_dmp, y_dmp, t_dmp]
    cov = [[variance, 0, 0], [0, variance, 0], [0, 0, variance]]
    x, y, t = np.random.multivariate_normal(mean, cov, 1).T
    # print("point returned from dmp_normal: ", (x, y, t))
    return x[0], y[0], t[0]


def sample_uniform(minx, miny, mint, maxx, maxy, maxt):

    x = np.random.uniform(low=minx, high=maxx, size=1)[0]
    y = np.random.uniform(low=miny, high=maxy, size=1)[0]
    t = np.random.uniform(low=mint, high=maxt, size=1)[0]

    return x, y, t


def ind_max(x):
    m = max(x)
    return x.index(m)


def get_state_reward(x, y, t, guiding_paths=None, weights=[1.0], obstacles=None, use_obstacle_cost=False,
                     obstacle_pot=1.0, epsilon=1/100000):

    if t < 0:
        return -1 * epsilon

    cost_total = 0.0
    for i in range(0, len(guiding_paths)):
        guiding_path = guiding_paths[i]
        weight = weights[i]

        min_index, _ = guiding_path.search(np.array([x, y, t]), 1)
        cost = sqrt((x - guiding_path.tree.data[min_index][0]) ** 2 + (y - guiding_path.tree.data[min_index][1]) ** 2)

        if use_obstacle_cost:
            obstacle_cost = 0
            if obstacles is not None:
                for obstacle in obstacles:
                    point = Point((x, y))

                    if obstacle.contains(point):
                        # print("returning neg reward")
                        return 1/epsilon, -1 * epsilon

                    else:

                        pol_ext = LinearRing(obstacle.exterior.coords)
                        d = pol_ext.project(point)
                        p = pol_ext.interpolate(d)
                        obst_potential_pt = list(p.coords)[0]
                        dist = sqrt((y - obst_potential_pt[1]) ** 2 +
                                    (x - obst_potential_pt[0]) ** 2)
                        obstacle_cost += obstacle_pot / ((dist + epsilon) ** 2)
                        # obstacle_cost += obstacle_pot / ((dist + epsilon) ** 2)
                cost += obstacle_cost

        else:
            for obstacle in obstacles:
                point = Point((x, y))

                if obstacle.contains(point):
                    # print("returning neg reward")
                    return -1 * epsilon

        cost_total += weight * cost
    return math.exp(-1 * cost_total)


def calculate_avg_distance(path):
    avg_distance = 0.0
    for i in range(0, len(path) - 1):
        avg_distance += math.sqrt(
            (path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2
            + (path[i][2] - path[i + 1][2]) ** 2)

    return avg_distance / len(path)



def calculate_discretised_edge_cost(origin, destination, guiding_paths, guiding_path_weights, edge_resolution,
                                    obstacles=[], use_obstacle_cost=True, obstacle_pot=1.0):

    cost = 0.0
    edge_length = math.sqrt((origin[0][0] - destination[0][0]) ** 2 + (origin[0][1] - destination[0][1]) ** 2 +
                            (origin[0][2] - destination[0][2]) ** 2)
    guiding_path_index = np.argmax(np.array(guiding_path_weights))
    guiding_path = guiding_paths[guiding_path_index]
    if edge_length <= edge_resolution:

        closest_pt_index, _ = guiding_path.search(np.array([destination[0][0], destination[0][1],
                                                            destination[0][2]]), 1)

        cost = math.sqrt((destination[0][0] - guiding_path.tree.data[closest_pt_index][0]) ** 2 +
                         (destination[0][1] - guiding_path.tree.data[closest_pt_index][1]) ** 2) * edge_length

    else:

        k = int(edge_length / edge_resolution)
        if k > 1:
            edge_points = [origin[0]]
            for i in range(1, (k + 1)):
                temp_x = ((k - i) * origin[0][0] + i * destination[0][0]) / k
                temp_y = ((k - i) * origin[0][1] + i * destination[0][1]) / k
                temp_t = ((k - i) * origin[0][2] + i * destination[0][2]) / k

                interm_pt = [temp_x, temp_y, temp_t]
                e = math.sqrt((temp_x - edge_points[-1][0]) ** 2 + (temp_y - edge_points[-1][1]) ** 2)
                edge_points.append(interm_pt)

                closest_pt_index, _ = guiding_path.search(np.array([temp_x, temp_y, temp_t]), 1)

                c = math.sqrt((interm_pt[0] - guiding_path.tree.data[closest_pt_index][0]) ** 2 +
                              (interm_pt[1] - guiding_path.tree.data[closest_pt_index][1]) ** 2)

                obstacle_cost = 0
                if use_obstacle_cost:
                    if obstacles is not None:
                        for obstacle in obstacles:
                            point = Point((temp_x, temp_y))
                            pol_ext = LinearRing(obstacle.exterior.coords)
                            d = pol_ext.project(point)
                            p = pol_ext.interpolate(d)
                            obst_potential_pt = list(p.coords)[0]
                            dist = sqrt((temp_y - obst_potential_pt[1]) ** 2 +
                                        (temp_x - obst_potential_pt[0]) ** 2)
                            obstacle_cost += obstacle_pot / ((dist + 0.0000001) ** 2)

                cost += (c + obstacle_cost) * e
        else:
            closest_pt_index, _ = guiding_path.search(np.array([destination[0][0], destination[0][1],
                                                                destination[0][2]]), 1)

            cost = math.sqrt((destination[0][0] - guiding_path.tree.data[closest_pt_index][0]) ** 2 +
                             (destination[0][1] - guiding_path.tree.data[closest_pt_index][1]) ** 2) * edge_length

    return cost

