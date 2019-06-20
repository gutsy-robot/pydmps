"""
Probablistic Road Map (PRM) Planner
author: Atsushi Sakai (@Atsushi_twi)
"""

# correct implementation in dijkstra's search method
from copy import deepcopy

import random
import math
from shapely.geometry import Point, mapping
from math import sqrt
from dynamic_kdtree import Node
from dynamic_kdtree import KDTree as DynamicKDTree
from union_find import UF
from ucb import UCB
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from statistics import mean
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import time


def sample_dmp_normal2D(x_dmp, y_dmp, variance=0.04):
    mean = [x_dmp, y_dmp]
    cov = [[variance, 0], [0, variance]]
    x, y = np.random.multivariate_normal(mean, cov, 1).T
    return x[0], y[0]


def sample_uniform2D(minx, miny, maxx, maxy):

    x = np.random.uniform(low=minx, high=maxx, size=1)[0]
    y = np.random.uniform(low=miny, high=maxy, size=1)[0]

    return x, y


def get_state_reward(x, y, guiding_paths=None, weights=[1.0], obstacles=None, use_obstacle_cost=True,
                     obstacle_pot=1.0, epsilon=1/100000):

    # returns state cost, reward

    cost_total = 0.0
    for i in range(0, len(guiding_paths)):
        guiding_path = guiding_paths[i]
        weight = weights[i]

        min_index, _ = guiding_path.search(np.array([x, y]), 1)
        cost = sqrt((x - guiding_path.tree.data[min_index][0]) ** 2 + (y - guiding_path.tree.data[min_index][1]) ** 2)

        if use_obstacle_cost:
            obstacle_cost = 0
            if obstacles is not None:
                for obstacle in obstacles:
                    point = Point((x, y))

                    if obstacle.contains(point):
                        return -1 * epsilon

                    else:

                        pol_ext = LinearRing(obstacle.exterior.coords)
                        d = pol_ext.project(point)
                        p = pol_ext.interpolate(d)
                        obst_potential_pt = list(p.coords)[0]
                        dist = sqrt((y - obst_potential_pt[1]) ** 2 +
                                    (x - obst_potential_pt[0]) ** 2)
                        obstacle_cost += obstacle_pot / ((dist + epsilon) ** 2)
                cost += obstacle_cost

        cost_total += weight * cost
    return math.exp(-1 * cost_total)


def calculate_avg_distance(path):
    avg_distance = 0.0
    for i in range(0, len(path) - 1):
        avg_distance += math.sqrt(
            (path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)

    return avg_distance / len(path)


def plan(start, goal, guiding_paths, obstacles, num_points=3000,
         reward_weights={'connectivity': 1.0, 'increemental': 1.0}, guiding_path_weights=[1.0],
         ucb=None, uniform_only=False, normal_only=False, plot_sampled=True, edge_resolution_factor=2.0,
         neighbor_radius_factor=2.0,  use_ucb=True, dynamic_radius=False,
         use_obstacle_cost=False, obstacle_pot=1.0, plot_roadmap=False, dmp_normal_cov=0.04,
         plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None, plt_roadmap=None, uniform_max=1.2,
         uniform_min=0.8):

    print("plan_ucb called..")
    print("total number of nodes in the roadmap should be: ", num_points)

    conn_comp_arr = []

    dmp = guiding_paths[0].tree.data
    dmp_x_min = np.min(dmp[:, 0])
    dmp_y_min = np.min(dmp[:, 1])

    dmp_x_max = np.max(dmp[:, 0])
    dmp_y_max = np.max(dmp[:, 1])

    avg_distance = calculate_avg_distance(dmp)

    neighbor_radius = neighbor_radius_factor * avg_distance
    edge_resolution = edge_resolution_factor * avg_distance
    print("average 3D distance in the DMP is: ", avg_distance)

    tree = DynamicKDTree()
    tree.add(start.points[0][0], start.points[0][1])
    tree.add(goal.points[0][0], goal.points[0][1])

    print("start and goal added to the Kdtree..")

    vertices = [start, goal]
    roadmap = {0: [], 1: []}

    print("uniform min: ", (uniform_min * dmp_x_min, uniform_min * dmp_y_min))
    print("uniform max: ", (uniform_max * dmp_x_max, uniform_max * dmp_y_max))

    # declare the union-find object
    uf = UF(2)

    if use_ucb:
        print("ucb is not None")
        mean1 = [ucb.values[0]]
        mean2 = [ucb.values[1]]

        print("mean1 and mean2 are: ", (mean1, mean2))

        ucb1 = []
        ucb2 = []

    increemental_reward = []
    connectivity_reward = []

    n_connected_components = len(vertices)
    node_number = 2
    normal_nodes = []
    uniform_nodes = []
    while len(vertices) < num_points:
        if use_ucb:
            arm, ucb_values = ucb.select_arm()
            if ucb_values is not None:
                ucb1.append(ucb_values[0])
                ucb2.append((ucb_values[1]))

        else:
            if uniform_only:
                arm = 0

            elif normal_only:
                arm = 1

        if arm == 0:
            x, y = sample_uniform2D(dmp_x_min * uniform_min, dmp_y_min * uniform_min,
                                    dmp_x_max * uniform_max, dmp_y_max * uniform_max)

        elif arm == 1:
            i = random.randint(0, len(dmp) - 1)
            x, y = sample_dmp_normal2D(dmp[i][0], dmp[i][1], variance=dmp_normal_cov)

        # get the dmp-time proximity reward.
        inc_reward = get_state_reward(x, y, guiding_paths=guiding_paths,
                                      weights=guiding_path_weights,
                                      obstacles=obstacles,
                                      use_obstacle_cost=use_obstacle_cost,
                                      obstacle_pot=obstacle_pot)

        reward = reward_weights['increemental'] * inc_reward

        increemental_reward.append(reward)

        # ensures whether the sampled state is feasible.
        if reward > 0:
            if plot_sampled:
                if arm == 0:
                    plt2d.plot(x, y, marker="+", color='b', markersize=2.0)
                    uniform_nodes.append(node_number)
                    node_number += 1

                elif arm == 1:
                    plt2d.plot(x, y, marker="+", color='r', markersize=2.0)
                    normal_nodes.append(node_number)
                    node_number += 1

            # add the 2D node to the union find object
            edges = []
            node = Node([([x, y, 0], [len(roadmap), 0.0, None, arm])])

            # might be useful to vary the distance as a function of num_points for asym. optimality
            # n_connected_components = uf.count()
            sampled_pt_id = len(roadmap)

            if dynamic_radius is not True:
                neighbors = tree.neighbors((x, y, 0), neighbor_radius)
            else:
                k = len(vertices) - 2
                print("value of k is: ", k)
                if k == 0 or k == 1:
                    r = neighbor_radius
                    print("neighbor radius is: ", neighbor_radius)
                else:
                    r = neighbor_radius * math.pow(math.log(k)/k, (1 / 2))
                    print("neighbor radius is: ", r)

                neighbors = tree.neighbors((x, y, 0), r)

            uf.add(sampled_pt_id)
            num_connections = 0

            for n in neighbors:
                neighbor_id = n[1][0]

                x_out = n[0][0]
                y_out = n[0][1]

                l = LineString([[x, y], [x_out, y_out]])
                intersect = False
                for obstacle in obstacles:
                    if l.intersects(obstacle):
                        intersect = True
                        break

                if not intersect:
                    edges.append(neighbor_id)
                    roadmap[neighbor_id].append(len(vertices))
                    uf.union(sampled_pt_id, neighbor_id)
                    num_connections += 1

            tree.add(node.points[0][0], node.points[0][1])
            roadmap[len(vertices)] = edges
            vertices.append(node)

            if reward_weights['connectivity'] > 0 and len(vertices) > 2:
                n_connected_components_new = uf.count()
                conn_comp_arr.append(n_connected_components_new)

                if n_connected_components_new < n_connected_components:
                    con_reward = reward_weights['connectivity'] * 0.7
                    reward += con_reward
                    connectivity_reward.append(con_reward)
                    n_connected_components = n_connected_components_new

                elif n_connected_components > n_connected_components:
                    con_reward = reward_weights['connectivity'] * 0.5
                    reward += con_reward
                    connectivity_reward.append(con_reward)
                    n_connected_components = n_connected_components_new

                else:
                    con_reward = 0.0
                    connectivity_reward.append(con_reward)

        if use_ucb:
            ucb.update(arm, reward)
            mean1.append(ucb.values[0])
            mean2.append(ucb.values[1])

    print("length of roadmap is: ", len(roadmap))

    if use_ucb:
        plt_mean_reward.plot(mean1, label='uniform')
        plt_mean_reward.plot(mean2, label='dmp_normal')
        plt_mean_reward.legend()

    increemental_reward = np.array(increemental_reward)
    connectivity_reward = np.array(connectivity_reward)
    print("max in the increemental reward array is: ", np.max(increemental_reward))
    print("min in the increemental reward array is: ", np.min(increemental_reward))
    print("mean incremental reward is: ", mean(increemental_reward))

    print("max in the connectivity reward array is: ", np.max(connectivity_reward))
    print("min in the connectivity reward array is: ", np.min(connectivity_reward))
    print("mean connectivity reward is: ", mean(connectivity_reward))

    if use_ucb:
        plt_ucb.plot(ucb1, label='uniform')
        plt_ucb.plot(ucb2, label='dmp normal')
        plt_ucb.legend()

    if use_ucb:
        plt.plot(conn_comp_arr)

    elif uniform_only:
        plt_connected.plot(conn_comp_arr, label='uniform')

    elif normal_only:
        plt_connected.plot(conn_comp_arr, label='normal')

    # if plot_roadmap:
    #     plot_road_map(roadmap, vertices, plt_roadmap)

    print("number of uniform nodes are: ", len(uniform_nodes))
    print("number of normal nodes are: ", len(normal_nodes))
    # plt.show()
    print("start edges: ", roadmap[0])
    print("goal edges: ", roadmap[1])
    return vertices, roadmap, ucb, edge_resolution


def dijkstra_planning(start, goal, road_map, vertices, guiding_paths=None, guiding_path_weights=None,
                      edge_resolution=None, use_obstacle_cost=True, obstacles=[],
                      obstacle_pot=1.0, plt2d=None, vmax=0.0,
                      vmin=2.0, delta_t=0.2):

    print("vmin is: ", vmin)
    print("vmax is: ", vmax)
    print("calling dijkstra's planning")
    openset, closedset = dict(), dict()
    openset[0] = start
    print("added start node to the openset")
    print("goal is: ", (goal.points[0][0][0], goal.points[0][0][1], goal.points[0][0][2]))

    while True:
        # print("loop agained")
        if not openset:
            print(openset)
            print("Cannot find path")
            goal.points[0][0][0] = current.points[0][0][0]
            goal.points[0][0][1] = current.points[0][0][1]
            goal.points[0][0][2] = current.points[0][0][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][1][2] = current.points[0][1][2]
            break

        min = None
        c_id = None
        for k, v in openset.items():
            cost = v.points[0][1][1]
            if min is None:
                min = cost
                c_id = k

            else:
                if cost < min:
                    min = cost
                    c_id = k

        current = openset[c_id]
        # print("cid is: ", c_id)

        if current.points[0][1][3] == -1:
            plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=4)

        elif current.points[0][1][3] == 0:
            plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=4)

        elif current.points[0][1][3] == 1:
            plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=4)

        else:
            print("POINT FOUND WHICH WON'T BE USING ")

        if c_id == 1:
            print("goal is found!")
            # print("c_id is: ", c_id)
            goal.points[0][1][2] = current.points[0][1][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][0][2] = current.points[0][0][2]
            goal.points[0][1][0] = current.points[0][1][0]
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current
        # print("current point is: ", (current.points[0][0][0], current.points[0][0][1], current.points[0][0][2]))

        if goal.points[0][0][0] == current.points[0][0][0] and goal.points[0][0][1] == current.points[0][0][1]:
            print("goal found in the 2D space")

        for i in range(0, (len(road_map[c_id]))):
            n_id = road_map[c_id][i]
            # if n_id == 1:
            # # print("goal node is one of the neighbors")

            if n_id in closedset:
                continue

            x = current.points[0][0][0]
            y = current.points[0][0][1]
            t = current.points[0][0][2]

            x_out = vertices[n_id].points[0][0][0]
            y_out = vertices[n_id].points[0][0][1]

            dist = math.sqrt((x_out - x) ** 2 + (y_out - y) ** 2)
            del_t_min = dist/vmax
            del_t_max = dist/vmin
            # print("delta t min is: ", del_t_min)
            # print("delta t max is: ", del_t_max)
            # print("number of time steps are: ", (del_t_max - del_t_min)/delta_t)
            t_out = t + del_t_min
            min_cost_node = None
            min_cost = math.inf
            while t + del_t_max > t_out:
                n = deepcopy(vertices[n_id])
                n.points[0][0][2] = t_out + del_t_min
                edge_cost = calculate_discretised_edge_cost(current.points[0], n.points[0],
                                                            guiding_paths, guiding_path_weights,
                                                            edge_resolution, obstacles=obstacles,
                                                            obstacle_pot=obstacle_pot,
                                                            use_obstacle_cost=use_obstacle_cost)
                # print("edge cost returned is: ", edge_cost)
                node_cost = edge_cost + current.points[0][1][1]
                n.points[0][1][1] = node_cost

                n.points[0][1][2] = c_id
                if node_cost < min_cost:
                    min_cost_node = n
                    min_cost = node_cost

                t_out += delta_t

            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].points[0][1][1] > min_cost_node.points[0][1][1]:
                    openset[n_id].points[0][0][2] = min_cost_node.points[0][0][2]
                    openset[n_id].points[0][1][1] = min_cost_node.points[0][1][1]
                    openset[n_id].points[0][1][2] = min_cost_node.points[0][1][2]
            else:
                openset[n_id] = min_cost_node
            # print("openset updated")

    rx, ry, rt, cost_array = [goal.points[0][0][0]], [goal.points[0][0][1]], [goal.points[0][0][2]], [goal.points[0]
                                                                                                      [1][1]]
    print("retracing the previous ids to get the path")
    pind = goal.points[0][1][2]
    path_cost = goal.points[0][1][1]
    while pind != -1:
        # print("entered inside while loop")
        n = closedset[pind]
        rx.append(n.points[0][0][0])
        ry.append(n.points[0][0][1])
        rt.append(n.points[0][0][2])
        cost_array.append(n.points[0][1][1])
        pind = n.points[0][1][2]

    return rx, ry, rt, path_cost, cost_array


def PRM_planning(sx, sy, gx, gy, obstacles=None, guiding_paths2d=None, guiding_paths3d=None,
                 dmp_vel=None, guiding_path_weights=[1.0],
                 reward_weights={'connectivity': 1.0, 'increemental': 0.1}, use_ucb=True, uniform_only=False,
                 normal_only=False, edge_resolution_factor=2.0, neighbor_radius_factor=1.0,
                 dynamic_radius=False, use_obstacle_cost=True, num_points=3000,
                 obstacle_pot=1.0, plot_sampled=False, plot_roadmap=False,
                 dmp_normal_cov=0.04, plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None,
                 plt_roadmap=None, uniform_max=1.2, uniform_min=0.8):

    # declare node as ((x, y, t), (id, cost, pind, distribution))
    start = Node([([sx, sy, 0], [0, 0, -1, -1])])

    print("time at goal is: ", guiding_paths3d[0].tree.data[-1][2])

    goal = Node([([gx, gy, 0], [1, 0, -1, -1])])

    print("start and goal nodes declared..")
    vel = []
    for v in dmp_vel:
        vel.append(math.sqrt(v[0] ** 2 + v[1] ** 2))

    vel = np.array(vel)

    v_max = max(vel)
    v_min = min(vel)

    print("vmax is: ", v_max)
    print("vmin is: ", v_min)

    if use_ucb:
        ucb = UCB()
        ucb.initialize(2)
    else:
        ucb = None

    # guiding paths given should be 2D
    vertices, roadmap, ucb_updated, edge_reso = plan(start, goal, guiding_paths2d, obstacles,
                                                     guiding_path_weights=guiding_path_weights, use_ucb=use_ucb,
                                                     ucb=ucb, reward_weights=reward_weights,
                                                     edge_resolution_factor=edge_resolution_factor,
                                                     neighbor_radius_factor=neighbor_radius_factor,
                                                     dynamic_radius=dynamic_radius,
                                                     use_obstacle_cost=use_obstacle_cost,
                                                     uniform_only=uniform_only, normal_only=normal_only,
                                                     num_points=num_points, obstacle_pot=obstacle_pot,
                                                     plot_sampled=plot_sampled, plot_roadmap=plot_roadmap,
                                                     dmp_normal_cov=dmp_normal_cov, plt2d=plt2d,
                                                     plt_mean_reward=plt_mean_reward,
                                                     plt_ucb=plt_ucb, plt_connected=plt_connected,
                                                     plt_roadmap=plt_roadmap, uniform_max=uniform_max,
                                                     uniform_min=uniform_min)

    # guiding paths given should be 3D
    rx, ry, rt, path_cost, cost_array = dijkstra_planning(
                                              start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                              guiding_paths=guiding_paths3d, edge_resolution=edge_reso,
                                              use_obstacle_cost=use_obstacle_cost, obstacles=obstacles,
                                              obstacle_pot=obstacle_pot,
                                              plt2d=plt2d, vmax=v_max, vmin=v_min)

    return rx, ry, rt, path_cost, cost_array


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
            for i in range(1, (k+1)):
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


