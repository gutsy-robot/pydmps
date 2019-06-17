import random
import math
from shapely.geometry import Point, mapping
from math import sqrt
from dynamic_kdtree import Node
from dynamic_kdtree import KDTree as DynamicKDTree
from union_find import UF
from ucb import UCB
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from utils import get_trajectory, check_collision, avoid_obstacles, sample_dmp_normal, sample_uniform, ind_max
from statistics import mean
from copy import deepcopy
from mpl_toolkits.mplot3d import axes3d
import time
import matplotlib.pyplot as plt
import numpy as np


def get_state_reward(x, y, t, guiding_paths=None, weights=[1.0], obstacles=None, use_obstacle_cost=True,
                     obstacle_pot=1.0, epsilon=1/100000):

    if t < 0:
        return 1/epsilon, -1 * epsilon

    cost_total = 0.0
    for i in range(0, len(guiding_paths)):
        # print("for i equal to: ", i)
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
                # print("total obstacle cost is: ", obstacle_cost)

        cost_total += weight * cost
    # print("returning from get state reward: ", (cost_total, math.exp(-1 * cost_total)))
    return cost_total, math.exp(-1 * cost_total)


def calculate_avg_distance(path):
    avg_distance = 0.0
    for i in range(0, len(path) - 1):
        avg_distance += math.sqrt(
            (path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2
            + (path[i][2] - path[i + 1][2]) ** 2)

    return avg_distance / len(path)


def plan(start, goal, guiding_paths, obstacles, v_max, v_min, num_points=3000,
         reward_weights={'connectivity': 1.0, 'increemental': 1.0}, guiding_path_weights=[1.0],
         ucb=None, uniform_only=False, normal_only=False, plot_sampled=True, edge_resolution_factor=2.0,
         neighbor_radius_factor=2.0, num_goal_pts=20, use_ucb=True, dynamic_radius=False,
         use_obstacle_cost=False, obstacle_pot=1.0, data_path="plots", plot_roadmap=False, dmp_normal_cov=0.04,
         plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None, plt_roadmap=None, uniform_max=1.2,
         uniform_min=0.8, uniform_max_t=1.5, plt_nodes=None, plt_reward=None):

    print("plan_ucb called..")
    print("total number of nodes in the roadmap should be: ", num_points)

    conn_comp_arr = []

    dmp_time = guiding_paths[0].tree.data
    time_reso = (dmp_time[1][2] - dmp_time[0][2])
    print("time reso is: ", time_reso)
    dmp_x_min = np.min(dmp_time[:, 0])
    dmp_y_min = np.min(dmp_time[:, 1])
    dmp_t_min = np.min(dmp_time[:, 2])

    dmp_x_max = np.max(dmp_time[:, 0])
    dmp_y_max = np.max(dmp_time[:, 1])
    dmp_t_max = np.max(dmp_time[:, 2])

    print("dmp_t_max is: ", dmp_t_max)
    avg_distance = calculate_avg_distance(dmp_time)

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
    print("uniform t max: ", (uniform_max_t * dmp_t_max))

    # declare the union-find object
    uf = UF(2)

    if ucb is not None:
        print("ucb is not None")
        mean1 = [ucb.values[0]]
        mean2 = [ucb.values[1]]

        print("mean1 and mean2 are: ", (mean1, mean2))

        ucb1 = []
        ucb2 = []

    # add points for reaching the goal(x,y) at different times
    for i in range(1, num_goal_pts):
        t = i * time_reso + goal.points[0][0][2]

        # the last -1 is to indicate the point is not from a particular distribution
        n = Node([([goal.points[0][0][0], goal.points[0][0][1], t],
                   [len(vertices), i * time_reso, -1, -1])])
        tree.add(n.points[0][0], n.points[0][1])
        vertices.append(n)
        roadmap[len(roadmap)] = []
        plt_nodes.scatter(goal.points[0][0][0], goal.points[0][0][1], t)

    print("length of vertices is: ", len(vertices))
    print("num_points are: ", num_points)

    increemental_reward = []
    connectivity_reward = []

    n_connected_components = len(vertices)
    node_number = num_goal_pts + 2
    normal_nodes = []
    uniform_nodes = []
    while len(vertices) < num_points:
        if ucb is not None:
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
            x, y, t = sample_uniform(dmp_x_min * uniform_min, dmp_y_min * uniform_min, 0,
                                     dmp_x_max * uniform_max, dmp_y_max * uniform_max, dmp_t_max * uniform_max_t)

        elif arm == 1:
            i = random.randint(0, len(dmp_time) - 1)
            x, y, t = sample_dmp_normal(dmp_time[i][0], dmp_time[i][1], dmp_time[i][2], variance=dmp_normal_cov)

        # get the dmp-time proximity reward.
        node_cost, inc_reward = get_state_reward(x, y, t, guiding_paths=guiding_paths,
                                                 weights=guiding_path_weights,
                                                 obstacles=obstacles,
                                                 use_obstacle_cost=use_obstacle_cost,
                                                 obstacle_pot=obstacle_pot)
        reward = reward_weights['increemental'] * inc_reward

        increemental_reward.append(reward)

        # ensures whether the sampled state is feasible.
        if reward > 0:
            # print("reward is greater than zero")
            if plot_sampled:
                # print("plot sampled is true")
                if arm == 0:
                    # if ucb is not None:
                    # print("(x, y) is: ", (x, y))
                    plt2d.plot(x, y, marker="+", color='b', markersize=2.0)
                    uniform_nodes.append(node_number)
                    node_number += 1
                    # print("point plotted")

                elif arm == 1:
                    # if ucb is not None:
                    # print("(x, y) is:  ", (x, y))
                    plt2d.plot(x, y, marker="+", color='r', markersize=2.0)
                    normal_nodes.append(node_number)
                    node_number += 1
                    # print("point plotted")
                # plt_nodes.scatter(x, y, t)

            # add the 2D node to the union find object
            edges = []
            node = Node([([x, y, t], [len(roadmap), node_cost, None, arm])])

            # might be useful to vary the distance as a function of num_points for asym. optimality
            # n_connected_components = uf.count()
            sampled_pt_id = len(roadmap) - num_goal_pts + 1

            if dynamic_radius is not True:
                neighbors = tree.neighbors((x, y, t), neighbor_radius)
                # print("length of neighbors is: ", len(neighbors))
            else:
                # print("dynamic radius is: ", neighbor_radius * (math.log(len(vertices))/len(vertices)) ** (1 /3) )
                k = len(vertices) - num_goal_pts - 1
                print("value of k is: ", k)
                if k == 0 or k == 1:
                    r = neighbor_radius
                    print("neighbor radius is: ", neighbor_radius)
                else:
                    r = neighbor_radius * math.pow(math.log(k)/k, (1 / 3))
                    # r = neighbor_radius
                    print("neighbor radius is: ", r)

                neighbors = tree.neighbors((x, y, t), r)

                # print("length of neighbors is: ", len(neighbors))

            uf.add(sampled_pt_id)
            num_connections = 0

            for n in neighbors:

                # CHANGE WAS MADE HERE
                if n[1][0] == 0:

                    # if the points are the original start or goal the id in UF and roadmap is the same
                    # neighbor_id = n[1][0]
                    neighbor_id = 0

                elif 1 <= n[1][0] <= num_goal_pts:
                    # neighbor_id = goal.points[0][1][0]
                    neighbor_id = 1
                    # if n[1][0] != 1:
                    #     print("got neighbor as one of the added goal points")
                    # else:
                    #     print("got neighor as the original goal pt.")

                else:
                    # otherwise account for the additional goals that were inserted into the roadmap
                    neighbor_id = n[1][0] - num_goal_pts + 1

                x_out = n[0][0]
                y_out = n[0][1]
                t_out = n[0][2]

                dist_2d = math.sqrt((x - x_out) ** 2 + (y - y_out) ** 2)

                if t_out > t:
                    vel = dist_2d / (t_out - t)
                    if v_min <= vel <= v_max:
                        l = LineString([[x, y], [x_out, y_out]])
                        intersect = False
                        for obstacle in obstacles:
                            if l.intersects(obstacle):
                                intersect = True
                                break

                        if not intersect:
                            edges.append(n[1][0])
                            uf.union(sampled_pt_id, neighbor_id)
                            num_connections += 1

                elif t_out < t:
                    vel = dist_2d / (t - t_out)
                    if v_min <= vel < v_max:
                        l = LineString([[x, y], [x_out, y_out]])
                        intersect = False
                        for obstacle in obstacles:
                            if l.intersects(obstacle):
                                intersect = True
                                break

                        if not intersect:
                            roadmap[n[1][0]].append(len(roadmap))
                            uf.union(sampled_pt_id, neighbor_id)
                            num_connections += 1

            tree.add(node.points[0][0], node.points[0][1])
            roadmap[len(vertices)] = edges
            vertices.append(node)
            if reward_weights['connectivity'] > 0 and len(vertices) > 1:
                n_connected_components_new = uf.count()
                # n_connected_components_new = len(uf.get_scc().keys())

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

        if ucb is not None:
            ucb.update(arm, reward)
            mean1.append(ucb.values[0])
            mean2.append(ucb.values[1])

    print("length of roadmap is: ", len(roadmap))

    if ucb is not None:
        plt_mean_reward.plot(mean1, label='uniform')
        plt_mean_reward.plot(mean2, label='dmp_normal')
        plt_mean_reward.legend()

        # plt_reward.plot(increemental_reward, label='increemental')
        # plt_reward.plot(connectivity_reward, label='connectivity')
        # plt_reward.legend()

    # plt_connectivity_reward.plot(connectivity_reward)
    increemental_reward = np.array(increemental_reward)
    connectivity_reward = np.array(connectivity_reward)
    print("max in the increemental reward array is: ", np.max(increemental_reward))
    print("min in the increemental reward array is: ", np.min(increemental_reward))
    print("mean incremental reward is: ", mean(increemental_reward))
    # print("plt mean reward done..")

    print("max in the connectivity reward array is: ", np.max(connectivity_reward))
    print("min in the connectivity reward array is: ", np.min(connectivity_reward))
    print("mean connectivity reward is: ", mean(connectivity_reward))

    # print("plt connectivity done..")

    if ucb is not None:
        plt_ucb.plot(ucb1, label='uniform')
        plt_ucb.plot(ucb2, label='dmp normal')
        plt_ucb.legend()

    if ucb is not None:
        # print("connected component array being plotted is: ", conn_comp_arr)
        plt.plot(conn_comp_arr)

    elif uniform_only:
        plt_connected.plot(conn_comp_arr, label='uniform')

    elif normal_only:
        plt_connected.plot(conn_comp_arr, label='normal')

    if plot_roadmap:
        plot_road_map(roadmap, vertices, plt_roadmap)

    print("number of uniform nodes are: ", len(uniform_nodes))
    print("number of normal nodes are: ", len(normal_nodes))
    # print("uniform node numbers are: ", uniform_nodes)
    # print("normal node numbers are: ", normal_nodes)

    return vertices, roadmap, ucb, edge_resolution


def dijkstra_planning(start, goal, road_map, vertices, guiding_paths=None, guiding_path_weights=None,
                      edge_resolution=None, use_discretised_cost=True, ucb_path=True, num_goal_pts=20,
                      use_obstacle_cost=True, obstacles=[], obstacle_pot=1.0, plt2d=None, plt_dij=None):

    print("calling dijkstra's planning")
    openset, closedset = dict(), dict()
    openset[0] = start
    print("added start node to the openset")

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
                c_id = v.points[0][1][0]
               
            else:
                if cost < min:
                    min = cost
                    c_id = v.points[0][1][0]

        current = openset[c_id]

        # show graph
        if ucb_path:
            if current.points[0][1][3] == -1:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=4)
                # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=4)

            elif current.points[0][1][3] == 0:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=4)
                # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=4)

            elif current.points[0][1][3] == 1:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=4)
                # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=4)

            else:
                print("POINT FOUND WHICH WON'T BE USING ")

            # ax.scatter(current.points[0][0][0], current.points[0][0][1], current.points[0][0][2])

        if c_id in range(1, (num_goal_pts + 1)):
            print("goal is found!")
            print("c_id is: ", c_id)
            goal.points[0][1][2] = current.points[0][1][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][0][2] = current.points[0][0][2]
            goal.points[0][1][0] = current.points[0][1][0]
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current
        # print("length of closedset is: ", len(closedset))

        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            # print("ni_id is: ", n_id)
            if 2 <= n_id < 49:
                print("SEARCH REACHES AUXILIARY TIME POINTS")

            node = deepcopy(vertices[n_id])
            # print("cost of the node in the vertices array when it was picked: ", node.points[0][1][1])
            if n_id in closedset:
                continue

            if use_discretised_cost:
                edge_cost = calculate_discretised_edge_cost(node.points[0], vertices[c_id].points[0],
                                                            guiding_paths, guiding_path_weights,
                                                            edge_resolution, obstacles=obstacles,
                                                            obstacle_pot=obstacle_pot,
                                                            use_obstacle_cost=use_obstacle_cost)

                node.points[0][1][1] = edge_cost + current.points[0][1][1]
                # print("after edge cost is added cost of the node in the vertices array is: ",
                #       vertices[n_id].points[0][1][1])
            else:
                # print("discretised cost is False")
                # node.points[0][1][1] += vertices[c_id].points[0][1][1]
                node.points[0][1][1] += current.points[0][1][1]

            node.points[0][1][2] = c_id

            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].points[0][1][1] > node.points[0][1][1]:

                    openset[n_id].points[0][1][1] = node.points[0][1][1]
                    openset[n_id].points[0][1][2] = c_id
            else:
                openset[n_id] = node

    rx, ry, rt, cost_array = [goal.points[0][0][0]], [goal.points[0][0][1]], [goal.points[0][0][2]], [goal.points[0]
                                                                                                      [1][1]]

    pind = goal.points[0][1][2]
    path_cost = goal.points[0][1][1]
    while pind != -1:
        # print("entered inside while loop")
        n = closedset[pind]
        rx.append(n.points[0][0][0])
        ry.append(n.points[0][0][1])
        rt.append(n.points[0][0][2])
        cost_array.append(n.points[0][1][1])
        # path_cost += n.points[0][1][1]
        pind = n.points[0][1][2]

    return rx, ry, rt, path_cost, cost_array


def PRM_planning(sx, sy, gx, gy, obstacles=None, guiding_paths=None, dmp_vel=None, guiding_path_weights=[1.0],
                 reward_weights={'connectivity': 1.0, 'increemental': 0.1}, use_ucb=True, uniform_only=False,
                 normal_only=False, edge_resolution_factor=2.0, neighbor_radius_factor=1.0, num_goal_pts=20,
                 dynamic_radius=False, use_obstacle_cost=True, data_path="plots", num_points=3000,
                 obstacle_pot=1.0, plot_sampled=False, plot_roadmap=False, use_discretised_cost=True,
                 dmp_normal_cov=0.04, plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None,
                 plt_roadmap=None, uniform_max=1.2, uniform_min=0.8, uniform_max_t=1.5, plt_dij=None, plt_nodes=None,
                 plt_reward=None):

    # declare node as ((x, y, t), (id, cost, pind, distribution))
    start = Node([([sx, sy, 0], [0, 0, -1, -1])])

    print("time at goal is: ", guiding_paths[0].tree.data[-1][2])

    goal = Node([([gx, gy, guiding_paths[0].tree.data[-1][2]], [1, 0, -1, -1])])

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

    vertices, roadmap, ucb_updated, edge_reso = plan(start, goal, guiding_paths, obstacles, v_max, v_min,
                                                     guiding_path_weights=guiding_path_weights, use_ucb=use_ucb,
                                                     ucb=ucb, reward_weights=reward_weights,
                                                     edge_resolution_factor=edge_resolution_factor,
                                                     neighbor_radius_factor=neighbor_radius_factor,
                                                     num_goal_pts=num_goal_pts, dynamic_radius=dynamic_radius,
                                                     use_obstacle_cost=use_obstacle_cost, data_path=data_path,
                                                     uniform_only=uniform_only, normal_only=normal_only,
                                                     num_points=num_points, obstacle_pot=obstacle_pot,
                                                     plot_sampled=plot_sampled, plot_roadmap=plot_roadmap,
                                                     dmp_normal_cov=dmp_normal_cov, plt2d=plt2d,
                                                     plt_mean_reward=plt_mean_reward,
                                                     plt_ucb=plt_ucb, plt_connected=plt_connected,
                                                     plt_roadmap=plt_roadmap, uniform_max=uniform_max,
                                                     uniform_min=uniform_min,
                                                     uniform_max_t=uniform_max_t, plt_nodes=plt_nodes,
                                                     plt_reward=plt_reward)

    rx, ry, rt, path_cost, cost_array = dijkstra_planning(
                                              start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                              guiding_paths=guiding_paths, edge_resolution=edge_reso,
                                              use_obstacle_cost=use_obstacle_cost, obstacles=obstacles,
                                              obstacle_pot=obstacle_pot, use_discretised_cost=use_discretised_cost,
                                              plt2d=plt2d, plt_dij=plt_dij, num_goal_pts=num_goal_pts)

    return rx, ry, rt, path_cost, cost_array


def calculate_discretised_edge_cost(origin, destination, guiding_paths, guiding_path_weights, edge_resolution,
                                    obstacles=[], use_obstacle_cost=True, obstacle_pot=1.0):

    cost = 0.0
    edge_length = math.sqrt((origin[0][0] - destination[0][0]) ** 2 + (origin[0][1] - destination[0][1]) ** 2 +
                            (origin[0][2] - destination[0][2]) ** 2)

    if edge_length <= edge_resolution:
        cost = destination[1][1]

    else:
        # print("edge length is more than edge resolution")
        guiding_path_index = np.argmax(np.array(guiding_path_weights))
        guiding_path = guiding_paths[guiding_path_index]
        k = int(edge_length / edge_resolution)
        if k > 1:
            # print("value of k is: ", k)
            for i in range(1, k):
                # print("doing for i equal to: ", i)
                temp_x = ((k - i) * origin[0][0] + i * destination[0][0]) / k
                temp_y = ((k - i) * origin[0][1] + i * destination[0][1]) / k
                temp_t = ((k - i) * origin[0][2] + i * destination[0][2]) / k

                interm_pt = [temp_x, temp_y, temp_t]
                # print("interm_pt is: ", interm_pt)

                closest_pt_index, _ = guiding_path.search(np.array([temp_x, temp_y, temp_t]), 1)
                # print("closest_pt_index is: ", closest_pt_index)
                # print("closest pt is: ", (guiding_path.tree.data[closest_pt_index][0],
                #                         guiding_path.tree.data[closest_pt_index][1]))

                c = math.sqrt((interm_pt[0] - guiding_path.tree.data[closest_pt_index][0]) ** 2 +
                              (interm_pt[1] - guiding_path.tree.data[closest_pt_index][1]) ** 2)
                # print("c is: ", c)

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

                cost += c + obstacle_cost
            cost += destination[1][1]

        else:
            # print("k is 1 ", k)
            cost = destination[1][1]

    return cost


def plot_road_map(roadmap, vertices, plt_roadmap):

    for k, v in roadmap.items():
        for node_id in v:
            plt_roadmap.arrow(vertices[k].points[0][0][0], vertices[k].points[0][0][1],
                              vertices[node_id].points[0][0][0] - vertices[k].points[0][0][0],
                              vertices[node_id].points[0][0][1] - vertices[k].points[0][0][1], shape='full', lw=0.2,
                              length_includes_head=True, head_width=.01)




