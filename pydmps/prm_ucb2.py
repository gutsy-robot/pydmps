import random
import math
from shapely.geometry import Point, mapping
from math import sqrt
from dynamic_kdtree import Node
from dynamic_kdtree import KDTree as DynamicKDTree
from union_find import UF
from ucb import UCB
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from utils import get_trajectory, check_collision, avoid_obstacles, sample_dmp_normal, sample_uniform, \
    ind_max, calculate_discretised_edge_cost
from statistics import mean
from copy import deepcopy
from mpl_toolkits.mplot3d import axes3d
import time
import matplotlib.pyplot as plt
import numpy as np


def get_state_reward(x, y, t, guiding_paths=None, weights=[1.0], obstacles=None, use_obstacle_cost=True,
                     obstacle_pot=1.0, epsilon=1/100000, state_reward_times=[]):

    # print("state_reward_times array in the start is: ", state_reward_times)
    st_time = time.time()

    if t < 0:
        return -1 * epsilon, state_reward_times

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
                        state_reward_times.append(time.time() - st_time)
                        return -1 * epsilon, state_reward_times

                    else:

                        pol_ext = LinearRing(obstacle.exterior.coords)
                        d = pol_ext.project(point)
                        p = pol_ext.interpolate(d)
                        obst_potential_pt = list(p.coords)[0]
                        dist = sqrt((y - obst_potential_pt[1]) ** 2 +
                                    (x - obst_potential_pt[0]) ** 2)
                        obstacle_cost += obstacle_pot / ((dist + epsilon) ** 2)
                cost += obstacle_cost

        else:
            for obstacle in obstacles:
                point = Point((x, y))

                if obstacle.contains(point):
                    # print("returning neg reward")
                    state_reward_times.append(time.time() - st_time)
                    return -1 * epsilon, state_reward_times

        cost_total += weight * cost

    state_reward_times.append(time.time() - st_time)
    return math.exp(-1 * cost_total), state_reward_times


def calculate_avg_distance(path):
    avg_distance = 0.0
    for i in range(0, len(path) - 1):
        avg_distance += math.sqrt(
            (path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2
            + (path[i][2] - path[i + 1][2]) ** 2)

    return avg_distance / len(path)


def plan(start, goal, guiding_paths, obstacles, v_max, v_min, num_points=3000,
         incremental_reward_weight=1.0, connectivity_reward_weight=1.0,  guiding_path_weights=[1.0],
         ucb=None, uniform_only=False, normal_only=False, plot_sampled=True, edge_resolution_factor=2.0,
         neighbor_radius_factor=2.0, num_goal_pts=20, use_ucb=True, dynamic_radius=False,
         use_obstacle_cost=False, obstacle_pot=1.0, plot_roadmap=False, dmp_normal_cov=0.04,
         plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None, plt_roadmap=None, uniform_max=1.2,
         uniform_min=0.8, uniform_max_t=1.5, plt_nodes=None, plt_fraction=None, lazy_collision_check=True,
         dmp_time_normal_cov=0.01, use_path_reduction_cost=True):

    call_st_time = time.time()
    st_reward_calc_times = []
    edge_collision_check_times = []

    print("plan_ucb called..")
    print("total number of nodes in the roadmap should be: ", num_points)
    # plt.show()
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
        mean3 = [ucb.values[2]]

        print("mean1 and mean2 are: ", (mean1, mean2))

        ucb1 = []
        ucb2 = []
        ucb3 = []

    # add points for reaching the goal(x,y) at different times
    for i in range(1, num_goal_pts):
        t = i * time_reso + goal.points[0][0][2]

        # the last -1 is to indicate the point is not from a particular distribution
        n = Node([([goal.points[0][0][0], goal.points[0][0][1], t],
                   [len(vertices), math.inf, -1, -1])])
        tree.add(n.points[0][0], n.points[0][1])
        vertices.append(n)
        roadmap[len(roadmap)] = []
        plt_nodes.scatter(goal.points[0][0][0], goal.points[0][0][1], t)

    print("length of vertices is: ", len(vertices))
    print("num_points are: ", num_points)

    increemental_reward = []
    connectivity_reward = []
    path_cost_reduction_reward_array = []

    n_connected_components = len(vertices)
    node_number = num_goal_pts + 2
    normal_nodes = []
    uniform_nodes = []

    # fraction_plotting_interval = num_points / 10
    pos = np.arange(10)
    bar_width = 0.25
    arms = ['uniform', 'normal_wide', 'normal_narrow']

    times_arm1 = 0
    times_arm2 = 0
    times_arm3 = 0

    arm1_fraction_arr = []
    arm2_fraction_arr = []
    arm3_fraction_arr = []
    runs_ucb = 0

    path_cost_array = []
    first_path_found = False

    while len(vertices) < num_points:
        # if len(vertices) == 160:
        #     temp = [v.points[0] for v in vertices[1: num_goal_pts + 1]]
        #     for v in temp:
        #         if v[0][0] != goal.points[0][0][0] and v[0][1] != goal.points[0][0][1]:
        #             print("found a goal node when the one of the goal pts doesn't have the same x and y "
        #                   "as goal. Len of ver"
        #                   "is: ",     len(vertices))
        #             print("node id is: ", v[1][0])
        #     # if runs_ucb != 0:

        if ucb is not None:
            arm, ucb_values = ucb.select_arm()
            runs_ucb += 1
            if ucb_values is not None:
                ucb1.append(ucb_values[0])
                ucb2.append(ucb_values[1])
                ucb3.append(ucb_values[2])

        else:
            if uniform_only:
                arm = 0

            elif normal_only:
                arm = 1

        if arm == 0:
            x, y, t = sample_uniform(dmp_x_min * uniform_min, dmp_y_min * uniform_min, 0,
                                     dmp_x_max * uniform_max, dmp_y_max * uniform_max, dmp_t_max * uniform_max_t)
            times_arm1 += 1

        elif arm == 1:
            k = random.randint(0, len(dmp_time) - 1)
            # x, y, t = dmp_time[k][0], dmp_time[k][1], dmp_time[k][2]
            x, y, t = sample_dmp_normal(dmp_time[k][0], dmp_time[k][1], dmp_time[k][2], variance=dmp_normal_cov,
                                        time_variance=dmp_time_normal_cov)
            times_arm2 += 1

        elif arm == 2:
            k = random.randint(0, len(dmp_time) - 1)
            # x, y, t = dmp_time[k][0], dmp_time[k][1], dmp_time[k][2]
            x, y, t = sample_dmp_normal(dmp_time[k][0], dmp_time[k][1], dmp_time[k][2], variance=dmp_normal_cov / 40,
                                        time_variance=dmp_time_normal_cov / 40)
            times_arm3 += 1

        if use_ucb:
            fraction_arm1 = times_arm1 / runs_ucb
            fraction_arm2 = times_arm2 / runs_ucb
            fraction_arm3 = times_arm3 / runs_ucb

            arm1_fraction_arr.append(fraction_arm1)
            arm2_fraction_arr.append(fraction_arm2)
            arm3_fraction_arr.append(fraction_arm3)

        # get the dmp-time proximity reward.
        inc_reward, st_reward_calc_times = get_state_reward(x, y, t, guiding_paths=guiding_paths,
                                                            weights=guiding_path_weights,
                                                            obstacles=obstacles,
                                                            use_obstacle_cost=use_obstacle_cost,
                                                            obstacle_pot=obstacle_pot,
                                                            state_reward_times=st_reward_calc_times)
        if not first_path_found:
            reward = incremental_reward_weight * inc_reward

        else:
            reward = 0.0

        increemental_reward.append(reward)

        # ensures whether the sampled state is feasible.
        if inc_reward > 0:
            if plot_sampled:
                if arm == 0:
                    plt2d.plot(x, y, marker="+", color='b', markersize=2.0)
                    uniform_nodes.append(node_number)
                    node_number += 1

                elif arm == 1:
                    plt2d.plot(x, y, marker="+", color='r', markersize=2.0)
                    normal_nodes.append(node_number)
                    node_number += 1

                elif arm == 2:
                    plt2d.plot(x, y, marker="+", color='g', markersize=2.0)
                    node_number += 1

            # add the 2D node to the union find object
            edges = []
            node = Node([([x, y, t], [len(roadmap), math.inf, None, arm])])

            # might be useful to vary the distance as a function of num_points for asym. optimality
            sampled_pt_id = len(roadmap) - num_goal_pts + 1

            if dynamic_radius is not True:
                neighbors = tree.neighbors((x, y, t), neighbor_radius)
            else:
                # print("dynamic radius is true")
                k = len(vertices) - num_goal_pts - 1
                # print("value of k is: ", k)
                if k == 0 or k == 1:
                    r = neighbor_radius
                    # print("neighbor radius is: ", neighbor_radius)
                else:
                    r = neighbor_radius * math.pow(math.log(k)/k, (1 / 3))
                    # print("neighbor radius is: ", r)

                neighbors = tree.neighbors((x, y, t), r)

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

                        # added fpr evaluating the lazy approach
                        if lazy_collision_check:

                            # in lazy collision check we will resuse the cost calculated during dijkstra
                            edges.append([n[1][0], False, math.inf, vel])
                            uf.union(sampled_pt_id, neighbor_id)

                        else:

                            intersect = False
                            for obstacle in obstacles:
                                coll_check_st_time = time.time()
                                if l.intersects(obstacle):
                                    edge_collision_check_times.append(time.time() - coll_check_st_time)
                                    intersect = True
                                    break

                            if not intersect:
                                edges.append([n[1][0], False, math.inf, vel])
                                uf.union(sampled_pt_id, neighbor_id)
                                num_connections += 1

                elif t_out < t:
                    vel = dist_2d / (t - t_out)
                    if v_min <= vel < v_max:
                        l = LineString([[x, y], [x_out, y_out]])

                        # added to evaluate the lazy approach
                        if lazy_collision_check:
                            roadmap[n[1][0]].append([len(roadmap), False, math.inf, vel])
                            uf.union(sampled_pt_id, neighbor_id)

                        else:
                            intersect = False

                            for obstacle in obstacles:
                                coll_check_st_time = time.time()
                                if l.intersects(obstacle):
                                    edge_collision_check_times.append(time.time() - coll_check_st_time)
                                    intersect = True
                                    break

                            if not intersect:
                                roadmap[n[1][0]].append([len(roadmap), False, math.inf, vel])
                                uf.union(sampled_pt_id, neighbor_id)
                                num_connections += 1

            tree.add(node.points[0][0], node.points[0][1])
            roadmap[len(vertices)] = edges
            vertices.append(node)

            # change in cost of optimal path reward
            if use_path_reduction_cost:
                path_cost_reduction_reward = 0.0
                st_uf_id = uf.find(0)
                goal_uf_id = uf.find(1)
                # roadmap_copy = deepcopy(roadmap)
                vertices_copy = deepcopy(vertices)
                start_copy = deepcopy(start)
                goal_copy = deepcopy(goal)
                if st_uf_id == goal_uf_id:
                    rx_, ry_, rt_, path_cost, cost_array_, path_indices_, path_found, velocities_ = dijkstra_planning(
                        start_copy, goal_copy, roadmap, vertices_copy, guiding_path_weights=guiding_path_weights,
                        guiding_paths=guiding_paths, edge_resolution=edge_resolution,
                        use_obstacle_cost=use_obstacle_cost, obstacles=obstacles,
                        obstacle_pot=obstacle_pot, use_discretised_cost=True, plt2d=None,
                        num_goal_pts=num_goal_pts, lazy_collision_check=lazy_collision_check)
                    if path_found:

                        if not first_path_found:
                            plt2d.plot(rx_, ry_, color="b", label="first feasible")
                            first_path_found = True
                            print("first path is found when length of vertices is: ", len(vertices))
                        else:
                            path_cost_reduction_reward =\
                                math.exp(max(path_cost_array[-1] - path_cost, 0.0) / path_cost_array[-1])
                            # reward += path_cost_reduction_reward

                        path_cost_reduction_reward_array.append(path_cost_reduction_reward)
                        path_cost_array.append(path_cost)

                    else:
                        path_cost_array.append(0.0)
                        path_cost_reduction_reward_array.append(path_cost_reduction_reward)
                else:
                    path_cost_array.append(0.0)
                    path_cost_reduction_reward_array.append(path_cost_reduction_reward)

            # we stop giving connectivity rewards when we find the first feasible path.
            con_reward = 0.0
            if connectivity_reward_weight > 0 and not first_path_found:
                n_connected_components_new = uf.count()

                conn_comp_arr.append(n_connected_components_new)

                if n_connected_components_new < n_connected_components:
                    con_reward = connectivity_reward_weight * 1.5
                    reward += con_reward
                    connectivity_reward.append(con_reward)
                    n_connected_components = n_connected_components_new

                elif n_connected_components > n_connected_components:
                    con_reward = connectivity_reward_weight * 1.0
                    reward += con_reward
                    connectivity_reward.append(con_reward)
                    n_connected_components = n_connected_components_new

                else:
                    con_reward = 0.0
                    connectivity_reward.append(con_reward)
            else:
                connectivity_reward.append(con_reward)

        if ucb is not None:
            ucb.update(arm, reward)
            mean1.append(ucb.values[0])
            mean2.append(ucb.values[1])
            mean3.append(ucb.values[2])

    print("length of roadmap is: ", len(roadmap))

    if ucb is not None:
        plt_mean_reward.plot(mean1, label='uniform')
        plt_mean_reward.plot(mean2, label='dmp_normal')
        plt_mean_reward.plot(mean3, label='dmp_normal_narrow')
        plt_mean_reward.legend()

    increemental_reward = np.array(increemental_reward)
    connectivity_reward = np.array(connectivity_reward)

    if ucb is not None:
        plt_ucb.plot(ucb1, label='uniform')
        plt_ucb.plot(ucb2, label='dmp normal')
        plt_ucb.plot(ucb3, label='narrow normal')
        plt_ucb.legend()

    if ucb is not None:
        plt_connected.plot(conn_comp_arr)

    elif uniform_only:
        plt_connected.plot(conn_comp_arr, label='uniform')

    elif normal_only:
        plt_connected.plot(conn_comp_arr, label='normal')

    if plot_roadmap:
        plot_road_map(roadmap, vertices, plt_roadmap)

    if use_ucb:
        fraction_plotting_interval = int(runs_ucb / 10)
        # print("fraction plotting interval is: ", fraction_plotting_interval)
        num_chosen_arm = []
        frac_arm1 = []
        frac_arm2 = []
        frac_arm3 = []

        for i in range(0, 10):
            ind = (i + 1) * fraction_plotting_interval - 1
            # print("ind is: ", ind)
            num_chosen_arm.append(ind + 1)
            frac_arm1.append(arm1_fraction_arr[ind])
            frac_arm2.append(arm2_fraction_arr[ind])
            frac_arm3.append(arm3_fraction_arr[ind])

        # print("fraction of arm1 is: ", frac_arm1)
        # print("fraction of arm2 is: ", frac_arm2)
        plt_fraction.bar(pos, frac_arm1, bar_width, color='blue', edgecolor='black')
        plt_fraction.bar(pos + bar_width, frac_arm2, bar_width, color='red', edgecolor='black')
        plt_fraction.bar(pos + 2 * bar_width, frac_arm3, bar_width, color='green', edgecolor='black')

        plt_fraction.set_xticks(pos)
        plt_fraction.set_xticklabels(num_chosen_arm)
        plt_fraction.set_xlabel('Number of Nodes', fontsize=16)
        plt_fraction.set_ylabel('Arm selection fraction', fontsize=16)
        plt_fraction.legend(arms, loc=2)

    total_process_time = time.time() - call_st_time

    return vertices, roadmap, ucb, edge_resolution, edge_collision_check_times, \
        st_reward_calc_times, total_process_time, path_cost_array, path_cost_reduction_reward_array, \
        connectivity_reward, increemental_reward


def dijkstra_planning(start, goal, road_map, vertices, guiding_paths=None, guiding_path_weights=None,
                      edge_resolution=None, use_discretised_cost=True, ucb_path=True, num_goal_pts=20,
                      use_obstacle_cost=True, obstacles=[], obstacle_pot=1.0, plt2d=None, lazy_collision_check=
                      True):

    # print("at the start of dijkstra planning cost of start node is: ", start.points[0][1][1])
    path_found = False
    # print("calling dijkstra's planning")
    openset, closedset = dict(), dict()
    openset[0] = start
    # print("added start node to the openset")

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
        # if ucb_path:
        #     if current.points[0][1][3] == -1:
        #         plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='y', markersize=4)
        #         # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=4)
        #
        #     elif current.points[0][1][3] == 0:
        #         plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=4)
        #         # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=4)
        #
        #     elif current.points[0][1][3] == 1:
        #         plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=4)
        #         # plt_dij.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=4)
        #
        #     elif current.points[0][1][3] == 2:
        #         plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=4)
        #
        #     else:
        #         print("POINT FOUND WHICH WON'T BE USING ")

        if c_id in range(1, (num_goal_pts + 1)):
            # print("goal is found!")
            # print("c_id is: ", c_id)
            goal.points[0][1][2] = current.points[0][1][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][0][2] = current.points[0][0][2]
            goal.points[0][1][0] = current.points[0][1][0]
            path_found = True
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        for i in range(len(road_map[c_id])):
            if lazy_collision_check:
                n_id = road_map[c_id][i][0]

            else:
                n_id = road_map[c_id][i][0]

            node = deepcopy(vertices[n_id])
            if n_id in closedset:
                continue

            if use_discretised_cost:
                if road_map[c_id][i][1] is True:
                    # print("edge cost already known, so using it..")
                    node.points[0][1][1] = road_map[c_id][i][2] + current.points[0][1][1]

                else:
                    edge_cost = calculate_discretised_edge_cost(node.points[0], vertices[c_id].points[0],
                                                                guiding_paths, guiding_path_weights,
                                                                edge_resolution, obstacles=obstacles,
                                                                obstacle_pot=obstacle_pot,
                                                                use_obstacle_cost=use_obstacle_cost)
                    node.points[0][1][1] = edge_cost + current.points[0][1][1]
                    # if lazy_collision_check:
                    road_map[c_id][i][2] = edge_cost
                    road_map[c_id][i][1] = True

                    # node_time = vertices[c_id].points[0][0][2]
                    # g = None
                    # for w in range(1, num_goal_pts + 1):
                    #     if vertices[w].points[0][0][2] >= node_time:
                    #         g = vertices[w].points[0]
                    #
                    # if g is not None:
                    #     cost_to_go = calculate_discretised_edge_cost(vertices[c_id].points[0], g,
                    #                                                  guiding_paths, guiding_path_weights,
                    #                                                  edge_resolution, obstacles=obstacles,
                    #                                                  obstacle_pot=obstacle_pot,
                    #                                                  use_obstacle_cost=use_obstacle_cost)
                    #
                    #     node.points[0][1][1] = edge_cost + cost_to_go + current.points[0][1][1]
                    #     # if lazy_collision_check:
                    #     road_map[c_id][i][2] = edge_cost + cost_to_go
                    #     road_map[c_id][i][1] = True
                    #
                    # else:
                    #     node.points[0][1][1] = math.inf
                    #     # if lazy_collision_check:
                    #     road_map[c_id][i][2] = math.inf
                    #     road_map[c_id][i][1] = True

            else:

                print("[ERROR]: Set discretised cost is True")

            node.points[0][1][2] = c_id

            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].points[0][1][1] > node.points[0][1][1]:

                    openset[n_id].points[0][1][1] = node.points[0][1][1]
                    openset[n_id].points[0][1][2] = node.points[0][1][2]
            else:
                openset[n_id] = node

    rx, ry, rt, cost_array = [goal.points[0][0][0]], [goal.points[0][0][1]], [goal.points[0][0][2]], [goal.points[0]
                                                                                                      [1][1]]

    pind = goal.points[0][1][2]
    path_cost = goal.points[0][1][1]
    path_indices = [goal.points[0][1][0]]
    velocities = []
    # print("Giving out final path from dijkstra's...")
    while pind != -1:
        n = closedset[pind]
        # if n == start:
        #     print("found start nude whose cost is: ", n.points[0][1][1])
        #     print("cost array when start was found is: ", cost_array)
        for ed in road_map[pind]:
            if ed[0] == path_indices[-1]:
                velocities.append(ed[3])

        rx.append(n.points[0][0][0])
        ry.append(n.points[0][0][1])
        rt.append(n.points[0][0][2])
        cost_array.append(n.points[0][1][1])
        path_indices.append(n.points[0][1][0])
        pind = n.points[0][1][2]
    velocities.append(0.0)
    return rx, ry, rt, path_cost, cost_array, path_indices, path_found, velocities


def PRM_planning(sx, sy, gx, gy, obstacles=None, guiding_paths=None, dmp_vel=None, guiding_path_weights=[1.0],
                 incremental_reward_weight=1.0, connectivity_reward_weight=1.0,
                 use_ucb=True, uniform_only=False,
                 normal_only=False, edge_resolution_factor=2.0, neighbor_radius_factor=1.0, num_goal_pts=20,
                 dynamic_radius=False, use_obstacle_cost=True, num_points=3000,
                 obstacle_pot=1.0, plot_sampled=False, plot_roadmap=False, use_discretised_cost=True,
                 dmp_normal_cov=0.04, plt2d=None, plt_mean_reward=None, plt_ucb=None, plt_connected=None,
                 plt_roadmap=None, uniform_max=1.2, uniform_min=0.8, uniform_max_t=1.5, plt_dij=None, plt_nodes=None,
                 plt_fraction=None, lazy_collision_check=True, dmp_normal_time_cov=0.01):

    # declare node as ((x, y, t), (node_id, cost, pind, distribution))
    # pind represents the id of the previous node on the optimal path.

    start = Node([([sx, sy, 0], [0, 0, -1, -1])])

    print("time at goal is: ", guiding_paths[0].tree.data[-1][2])

    # goal = Node([([gx, gy, guiding_paths[0].tree.data[-1][2]], [1, 0, -1, -1])])
    goal = Node([([gx, gy, guiding_paths[0].tree.data[-1][2]], [1, math.inf, -1, -1])])

    print("start and goal nodes declared..")
    vel = []
    for v in dmp_vel:
        vel.append(math.sqrt(v[0] ** 2 + v[1] ** 2))

    vel = np.array(vel)
    # v_max = 10000000
    v_max = max(vel)
    # v_min = min(vel)
    v_min = 0.0
    print("vmax is: ", v_max)
    print("vmin is: ", v_min)

    if use_ucb:
        ucb = UCB()
        ucb.initialize(3)
    else:
        ucb = None

    vertices, roadmap, ucb_updated, edge_reso, edge_check_time_sampling, reward_calc_time, total_sampling_time,\
        path_cost_array, path_cost_reduction_reward_array, connectivity_reward, increemental_reward= \
        plan(start, goal, guiding_paths, obstacles, v_max, v_min, guiding_path_weights=guiding_path_weights,
             use_ucb=use_ucb, ucb=ucb, incremental_reward_weight=incremental_reward_weight,
             connectivity_reward_weight=connectivity_reward_weight,
             edge_resolution_factor=edge_resolution_factor, neighbor_radius_factor=neighbor_radius_factor,
             num_goal_pts=num_goal_pts, dynamic_radius=dynamic_radius, use_obstacle_cost=use_obstacle_cost,
             uniform_only=uniform_only, normal_only=normal_only, num_points=num_points, obstacle_pot=obstacle_pot,
             plot_sampled=plot_sampled, plot_roadmap=plot_roadmap, dmp_normal_cov=dmp_normal_cov, plt2d=plt2d,
             plt_mean_reward=plt_mean_reward, plt_ucb=plt_ucb, plt_connected=plt_connected,
             plt_roadmap=plt_roadmap, uniform_max=uniform_max, uniform_min=uniform_min, uniform_max_t=uniform_max_t,
             plt_nodes=plt_nodes, plt_fraction=plt_fraction,  lazy_collision_check=lazy_collision_check,
             dmp_time_normal_cov=dmp_normal_time_cov)

    if lazy_collision_check:
        plan_st_time = time.time()
        collision_free_path = False
        path_found = True
        while not collision_free_path and path_found:
            rx, ry, rt, path_cost, cost_array, path_indices, path_found, velocities = dijkstra_planning(
                start, goal, roadmap, vertices,
                guiding_path_weights
                =guiding_path_weights,
                guiding_paths=guiding_paths,
                edge_resolution=edge_reso,
                use_obstacle_cost=
                use_obstacle_cost,
                obstacles=obstacles,
                obstacle_pot=obstacle_pot,
                use_discretised_cost=
                use_discretised_cost,
                plt2d=plt2d,
                num_goal_pts=num_goal_pts,
                lazy_collision_check=lazy_collision_check)

            if path_found:
                is_free, roadmap = check_path(rx, ry, obstacles, roadmap, path_indices)
                collision_free_path = is_free

    else:
        plan_st_time = time.time()
        rx, ry, rt, path_cost, cost_array, path_indices, path_found,\
            velocities = dijkstra_planning(start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                           guiding_paths=guiding_paths, edge_resolution=edge_reso, use_obstacle_cost=
                                           use_obstacle_cost, obstacles=obstacles, obstacle_pot=obstacle_pot,
                                           use_discretised_cost=use_discretised_cost, plt2d=plt2d, num_goal_pts=
                                           num_goal_pts, lazy_collision_check=lazy_collision_check)

    dijkstra_time = time.time() - plan_st_time

    print("total sampling time is: ", total_sampling_time)
    print("total time taken by dijkstra's is: ", dijkstra_time)

    return rx, ry, rt, path_cost, cost_array, edge_check_time_sampling, reward_calc_time, total_sampling_time, \
        dijkstra_time, velocities, v_max, path_cost_array, path_found, path_cost_reduction_reward_array, \
        connectivity_reward, increemental_reward


def plot_road_map(roadmap, vertices, plt_roadmap):

    for k, v in roadmap.items():
        for node_id in v:
            plt_roadmap.arrow(vertices[k].points[0][0][0], vertices[k].points[0][0][1],
                              vertices[node_id].points[0][0][0] - vertices[k].points[0][0][0],
                              vertices[node_id].points[0][0][1] - vertices[k].points[0][0][1], shape='full', lw=0.2,
                              length_includes_head=True, head_width=.01)


def check_path(path_x, path_y, obstacles, roadmap, path_indices):

    print("=========")
    print("check path called..")

    is_free_path = True
    for i in range(0, len(path_x) - 1):
        l = LineString([[path_x[i+1], path_y[i+1]], [path_x[i], path_y[i]]])
        out_edges = roadmap[path_indices[i + 1]]

        for k in range(0, len(out_edges)):
            tup = out_edges[k]
            # print("tup is: ", tup)
            if tup[0] == path_indices[i]:
                # print("found the edge with id: ", path_indices[i])
                if tup[1] is False:
                    # no need to do this when using the lazy approach
                    intersect = False
                    for obstacle in obstacles:
                        if l.intersects(obstacle):
                            intersect = True
                            is_free_path = False
                            break

                    if intersect:
                        print("found an edge that intersects...")
                        source_node = path_indices[i + 1]

                        # see if this requires deepcopy
                        edges = roadmap[source_node]

                        # print("length of the outgoing edges array before deletion: ", len())
                        # remove the outgoing colling edge
                        for s in range(0, len(edges)):
                            ind = edges[s]
                            if ind[0] == path_indices[i]:
                                del edges[s]
                                return is_free_path, roadmap

                    else:
                        print("edge status before ammending is: ", roadmap[path_indices[i+1]][k][1])
                        tup[1] = True
                        print("edge status after ammending is: ", roadmap[path_indices[i + 1]][k][1])

                    print("the edge is collision free")

                # else:
                #     print("edge has been previously checked and is collision free")

    if not is_free_path:
        print("path found by dijkstra is not collision free will replan")

    return is_free_path, roadmap
