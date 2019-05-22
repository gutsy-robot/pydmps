import random
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from shapely.geometry import Point, mapping
from math import sqrt, ceil, floor
from utils import get_trajectory, check_collision, avoid_obstacles, sample_dmp_normal, sample_uniform, ind_max
from dmp_discrete import DMPs_discrete
from kdtree import Node, KDTree
from union_find import UF
from ucb import UCB
import time

N_SAMPLE = 500          # number of sample_points
N_KNN = 80              # number of edge from one sampled point
MAX_EDGE_LEN = 30.0     # [m] Maximum edge length

show_animation = True

fig1 = plt.figure(1)
plt2d = fig1.add_subplot(111)
fig1.suptitle('Path Plot', fontsize=24)

fig2 = plt.figure(2)
plt_mean_reward = fig2.add_subplot(111)
fig2.suptitle('Mean Reward', fontsize=24)

fig3 = plt.figure(3)
plt_increemental_reward = fig3.add_subplot(111)
fig3.suptitle('Increemental Reward', fontsize=24)

fig4 = plt.figure(4)
plt_connectivity_reward = fig4.add_subplot(111)
fig4.suptitle('Connectivity Reward', fontsize=24)

fig5 = plt.figure(5)
plt_ucb = fig5.add_subplot(111)
fig5.suptitle('UCB Values', fontsize=24)

fig6 = plt.figure(6)
plt_connected = fig6.add_subplot(111)
fig6.suptitle('Number of connected components', fontsize=24)


def get_state_reward(x, y, t, guiding_paths=None, weights=[1.0], obstacles=None):

    if t < 0:
        return 1/100000

    cost_total = 0.0
    for i in range(0, len(guiding_paths)):
        guiding_path = guiding_paths[i]
        weight = weights[i]

        d = []

        for pt in guiding_path:
            distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2 + (t - pt[2]) ** 2)
            d.append(distance)

        d = np.array(d)

        min_index = np.argmin(d)
        cost = sqrt((x - guiding_path[min_index][0]) ** 2 + (y - guiding_path[min_index][1]) ** 2
                    + (t - guiding_path[min_index][2]) ** 2)

        obstacle_cost = 0
        if obstacles is not None:
            for obstacle in obstacles:
                point = Point((pt[0], pt[1]))

                if obstacle.contains(point):
                    return 1/100000

                else:

                    pol_ext = LinearRing(obstacle.exterior.coords)
                    d = pol_ext.project(point)
                    p = pol_ext.interpolate(d)
                    obst_potential_pt = list(p.coords)[0]
                    dist = sqrt((y - obst_potential_pt[1]) ** 2 +
                                (x - obst_potential_pt[0]) ** 2)
                    obstacle_cost += 100 / ((dist + 0.0000001) ** 2)

        # cost += obstacle_cost
        cost_total += weight * cost

    return 1/cost_total


def plan_ucb(start, goal, guiding_paths, obstacles, v_max, v_min, num_points=10000,
             reward_weights={'connectivity': 1.0, 'increemental': 1.0}, guiding_path_weights=[1.0],
             ucb=None):

    print("plan_ucb called..")
    print("total number of nodes in the roadmap should be: ", num_points)

    conn_comp_arr = []

    dmp_time = guiding_paths[0]
    time_reso = dmp_time[1][2] - dmp_time[0][2]
    dmp_x_min = np.min(dmp_time[:, 0])
    dmp_y_min = np.min(dmp_time[:, 1])
    dmp_t_min = np.min(dmp_time[:, 2])

    dmp_x_max = np.max(dmp_time[:, 0])
    dmp_y_max = np.max(dmp_time[:, 1])
    dmp_t_max = np.max(dmp_time[:, 2])

    avg_distance = 0.0
    for i in range(0, len(dmp_time) - 1):
        avg_distance += math.sqrt((dmp_time[i][0] - dmp_time[i+1][0]) ** 2 + (dmp_time[i][1] - dmp_time[i+1][1]) ** 2
                                  + (dmp_time[i][2] - dmp_time[i+1][2]) ** 2)

    avg_distance /= len(dmp_time)
    print("average 3D distance in the DMP is: ", avg_distance)

    print("min for uniform distribution is: ", (dmp_x_min, dmp_y_min, dmp_t_min))
    print("max for uniform distribution is: ", (dmp_x_max, dmp_y_max, dmp_t_max))

    tree = KDTree()
    tree.add(start.points[0][0], start.points[0][1])
    tree.add(goal.points[0][0], goal.points[0][1])

    print("start and goal added to the Kdtree..")

    vertices = [start, goal]
    roadmap = {0: [], 1: []}
    uf = UF(2)
    mean1 = [ucb.values[0]]
    mean2 = [ucb.values[1]]

    ucb1 = []
    ucb2 = []

    # add points for reaching the goal(x,y) at different times
    num_goal_pts = 20
    for i in range(1, num_goal_pts):
        t = i * time_reso + goal.points[0][0][2]

        # the last -1 is to indicate the point is not from a particular distribution
        n = Node([([goal.points[0][0][0], goal.points[0][0][1], t],
                  [len(vertices), i * time_reso, -1, -1])])
        tree.add(n.points[0][0], n.points[0][1])
        vertices.append(n)
        roadmap[len(roadmap)] = []

    print("length of vertices is: ", len(vertices))
    print("num_points are: ", num_points)
    increemental_reward = []
    connectivity_reward = []
    while len(vertices) < num_points:
        arm, ucb_values = ucb.select_arm()
        if ucb_values is not None:
            ucb1.append(ucb_values[0])
            ucb2.append((ucb_values[1]))

        if arm == 0:
            x, y, t = sample_uniform(0, 0, 0,
                                    dmp_x_max, dmp_y_max, dmp_t_max)

        else:
            i = random.randint(0, len(dmp_time) - 1)
            x, y, t = sample_dmp_normal(dmp_time[i][0], dmp_time[i][1], dmp_time[i][2])

        # get the dmp-time proximity reward.

        reward = reward_weights['increemental'] * \
            get_state_reward(x, y, t, guiding_paths=guiding_paths, weights=guiding_path_weights,
                             obstacles=obstacles)

        increemental_reward.append(reward)
        # prev_count = None
        # prev_id = None
        # ensures whether the sampled state is feasible.
        if reward > 0 or reward_weights['connectivity'] > 0.0:

            # add the 2D node to the union find object
            edges = []
            node = Node([([x, y, t], [len(roadmap), 1/reward, None, arm])])
            # might be useful to vary the distance as a function of num_points for asym. optimality
            n_connected_components = uf.count()
            sampled_pt_id = len(roadmap) - num_goal_pts + 1

            neighbors = tree.neighbors((x, y, t), avg_distance)

            # print("number of neigbours are: ", len(neighbors))
            # print("----------")
            # if len(neighbors) > 0:
            #     print("number of neighbors are: ", len(neighbors))
            #     print("before information..")
            #     print("number of nodes are: ", uf.get_num_nodes())
            #     print("self.count is: ", uf.count())
            #     print("number of unique ids in the uf is: ", len(set(uf._id)))

            # print("adding node to uf whose id length is.......................: ", len(uf._id))
            # print("id for node being added.............................: ", sampled_pt_id)
            uf.add(sampled_pt_id)

            for n in neighbors:
                if n[1][0] == 0 or n[1][0] == 1:

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
                        intersect = False
                        for obstacle in obstacles:
                            if l.intersects(obstacle):
                                intersect = True
                                break

                        if not intersect:
                            edges.append(n[1][0])
                            uf.union(sampled_pt_id, neighbor_id)

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
                            # print("union called")
                            uf.union(sampled_pt_id, neighbor_id)

                # actual_num_scc = len(uf.get_scc().keys())
                #
                # if uf.count() != actual_num_scc:
                #     print("neighbor id is: ", neighbor_id)
                #     print("sampled_pt id is: ", sampled_pt_id)
                #
                #     print("DIFFERENCE FOUND...........")
                #     print("actual number of scc; ", actual_num_scc)
                #     print("N is:  ", uf.count())
                #     print("id array is: ", uf._id)
                #     print("rank is: ", uf._rank)
                #
                #     print("prev count is: ", prev_count)
                #     print("prev_id is: ", prev_id)
                #     return

                # if uf.count() != len(set(uf._id)):
                #     print("difference found!!")
                #
                #     print("sampled_pt id is: ", sampled_pt_id)
                #     print("neighbor id is: ", neighbor_id)
                #
                #     print("sampled_pt uf id is: ", uf.find(sampled_pt_id))
                #     print("neighbor pt uf id is: ", uf.find(neighbor_id))
                #
                #     temp = prev_id
                #     p = neighbor_id
                #     while p != temp[p]:
                #         p = temp[p] = temp[temp[p]]
                #
                #     print("neighbor id in previous was: ", p)
                #
                #     temp = prev_id
                #     p = sampled_pt_id
                #
                #     while p != temp[p]:
                #         p = temp[p] = temp[temp[p]]
                #
                #     print("sampled_pt id in previous was: ", p)
                #
                #     print("previous count was: ", prev_count)
                #     print("self.count is: ", uf.count())
                #
                #     print("number of unique ids prev: ", len(set(prev_id)))
                #     print("number of unique ids in the uf is: ", len(set(uf._id)))
                #
                #     print("prev id are: ", prev_id)
                #     print("id are: ", uf._id)
                #
                #     print("id in the roadmap is: ", n[1][0])
                #     print("edges from the node are: ", roadmap[n[1][0]])
                #     print("length of vertices is: ", len(vertices))
                #     print("length of ids is: ", len(uf._id))
                #
                #     return
                #
                # prev_count = uf.count()
                # prev_id = uf._id

            tree.add(node.points[0][0], node.points[0][1])
            roadmap[len(vertices)] = edges
            vertices.append(node)
            # n_connected_components_new = uf.count()
            n_connected_components_new = len(uf.get_scc().keys())

            conn_comp_arr.append(n_connected_components_new)
            # print("difference between new num connected vs old is: ", (n_connected_components_new -
            #                                                            n_connected_components))
            if n_connected_components_new < n_connected_components:
                con_reward = reward_weights['connectivity'] * 1.0
                reward += con_reward
                connectivity_reward.append(con_reward)

            elif n_connected_components > n_connected_components:
                con_reward = reward_weights['connectivity'] * 1.5
                reward *= con_reward
                connectivity_reward.append(con_reward)

            else:
                con_reward = 0.0
                connectivity_reward.append(con_reward)

        ucb.update(arm, reward)
        mean1.append(ucb.values[0])
        mean2.append(ucb.values[1])

    print("length of roadmap is: ", len(roadmap))

    plt_mean_reward.plot(mean1, 'bo', label='uniform')
    plt_mean_reward.plot(mean2, 'r+', label='dmp_normal')
    plt_mean_reward.legend()

    plt_increemental_reward.plot(increemental_reward)
    plt_connectivity_reward.plot(connectivity_reward)
    increemental_reward = np.array(increemental_reward)

    print("max in the increemental reward array is: ", np.max(increemental_reward))
    print("min in the increemental reward array is: ", np.min(increemental_reward))
    print("plt mean reward done..")

    plt_ucb.plot(ucb1, label='uniform')
    plt_ucb.plot(ucb2, label='dmp normal')
    plt_ucb.legend()

    plt_connected.plot(conn_comp_arr)

    return vertices, roadmap, ucb


def dijkstra_planning(start, goal, road_map, vertices):

    print("calling dijkstra's planning")
    openset, closedset = dict(), dict()
    openset[0] = start
    print("added start node to the openset")

    while True:
        # print("loop agained")
        if not openset:
            print(openset)
            print("Cannot find path")
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
        if show_animation and len(closedset.keys()) % 2 == 0:
            if current.points[0][1][3] == -1:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g')

            elif current.points[0][1][3] == 0:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b')

            elif current.points[0][1][3] == 1:
                plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r')

            # ax.scatter(current.points[0][0][0], current.points[0][0][1], current.points[0][0][2])

        if c_id in range(1, 21):
            print("goal is found!")
            print("c_id is: ", c_id)
            goal.points[0][1][2] = current.points[0][1][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][0][2] = current.points[0][0][2]

            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current
        # print("length of closedset is: ", len(closedset))

        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            # print("ni_id is: ", n_id)
            if n_id in range(1, 21):
                print("goal pt reached..")

            node = vertices[n_id]
            node.points[0][1][1] += vertices[c_id].points[0][1][1]
            node.points[0][1][2] = c_id

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].points[0][1][1] > node.points[0][1][1]:
                    openset[n_id].points[0][1][1] = node.points[0][1][1]
                    openset[n_id].points[0][1][2] = c_id
            else:
                openset[n_id] = node

    # generate final course

    rx, ry, rt = [goal.points[0][0][0]], [goal.points[0][0][1]], [goal.points[0][0][2]]
    pind = goal.points[0][1][2]
    while pind != -1:
        # print("entered inside while loop")
        n = closedset[pind]
        rx.append(n.points[0][0][0])
        ry.append(n.points[0][0][1])
        rt.append(n.points[0][0][2])
        pind = n.points[0][1][2]

    return rx, ry, rt


def PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=None, dmp_vel=None, guiding_path_weights=[1.0],
                 reward_weights={'connectivity': 1.0, 'increemental': 0.000001}):

    """

    :param sx: x coordinate of start
    :param sy: y coordinate of start
    :param gx: x coordinate of goal
    :param gy: y coordinate of goal
    :param obstacles: obstacles arry
    :param guiding_paths: array of guiding paths, primary dmp path should be at index 0
    :param guiding_path_weights: weight for different cost from different guiding paths
    :param dmp_vel: velocity at different points for the primary dmp path
    :param reward_weights: relative importance of the different component costs
    :return: space-time path for reaching the goal
    """

    # declare node as ((x, y, t), (id, cost, pind))
    start = Node([([sx, sy, 0], [0, 0, -1, -1])])

    print("time at goal is: ", guiding_paths[0][-1][2])

    goal = Node([([gx, gy, guiding_paths[0][-1][2]], [1, 0, -1, -1])])

    print("start and goal nodes declared..")
    vel = []
    for v in dmp_vel:
        vel.append(math.sqrt(v[0] ** 2 + v[1] ** 2))

    vel = np.array(vel)

    v_max = max(vel)
    v_min = min(vel)

    print("vmax is: ", v_max)
    print("vmin is: ", v_min)

    ucb = UCB()
    ucb.initialize(2)

    vertices, roadmap, ucb_updated = plan_ucb(start, goal, guiding_paths, obstacles, v_max, v_min,
                                              guiding_path_weights=guiding_path_weights,
                                              ucb=ucb, reward_weights=reward_weights)
    rx, ry, rt = dijkstra_planning(start, goal, roadmap, vertices)

    return rx, ry, rt, ucb_updated


def main(path_x=None, path_y=None):
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0 * 0.01
    sy = 10.0 * 0.01

    gx = 75.0 * 0.01
    gy = 70.0 * 0.01

    plot_paths = []
    legend_key = []

    z = np.zeros((1, len(path_x)))

    # orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z, '--', linewidth=2)
    plt2d.plot(path_x, path_y, '--', linewidth=2, label='demonstration')
    # plot_paths.append(orig_path)
    # ax.scatter(path_x[0], path_y[0], z[0][0])
    plt2d.scatter(path_x[0], path_y[0])
    plt2d.annotate("st. original", (path_x[0], path_y[0]))
    # ax.scatter(path_x[-1], path_y[-1], z[0][-1])
    plt2d.scatter(path_x[-1], path_y[-1])
    plt2d.annotate("f original.", (path_x[-1], path_y[-1]))

    # ax.scatter(sx, sy, 0.0)
    plt2d.scatter(sx, sy)
    plt2d.annotate("new_st", (sx, sy))

    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    # coords = [(150.0, 120.0), (150.0, 140.0), (160.0, 140.0), (160.0, 120.0)]
    coords = [(50.0 * 0.01, 30.0 * 0.01), (50.0 * 0.01, 40.0 * 0.01), (60.0 * 0.01, 40.0 * 0.01),
              (60.0 * 0.01, 30.0 * 0.01)]
    poly1 = Polygon(coords)

    coords2 = [(25.0, 15.0), (25.0, 25.0), (35.0, 25.0), (35.0, 15.0)]
    poly2 = Polygon(coords2)

    obstacles = [poly1]
    print("[INFO]: obstacles created")

    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        plt2d.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    plt2d.plot(sx, sy, "xr")
    plt2d.plot(gx, gy, "xb")
    plt2d.grid(True)
    plt2d.axis("equal")
    print("[INFO]: Plotted obstacles and start and end pts..")

    dmp = DMPs_discrete(n_dmps=2, n_bfs=100, dt=0.01, run_time=1.0)

    dmp.imitate_path(y_des=np.array([path_x, path_y]))
    dmp.y0[0] = sx
    dmp.y0[1] = sy
    # change the scale of the movement

    # TODO: apply goal_check and goal_alteration
    dmp.goal[0] = gx
    dmp.goal[1] = gy

    y_track_nc, dy_track_nc, ddy_track_nc, s = dmp.rollout()
    print("[INFO]: trajectory rolled out successfully..")
    print("shape of y_track_nc is: ", y_track_nc.shape)

    y_track_nc_time = []
    print("total time is: ", len(y_track_nc) * dmp.dt)
    for i in range(0, len(y_track_nc)):
        y_track_nc_time.append((i + 1) * dmp.dt)

    print("end time in the time array is: ", y_track_nc_time[-1])
    # print("the trajectory rolled out by the dmp is: ", y_track_nc)
    y_track_nc_time = np.array([y_track_nc_time])
    y_track_nc_x = np.array(y_track_nc[:, 0])
    y_track_nc_y = np.array(y_track_nc[:, 1])

    # ax.scatter(y_track_nc_x, y_track_nc_y, y_track_nc_time)
    # # ax.plot_wireframe(y_track_nc_x, y_track_nc_y, y_track_nc_time)
    # plot = ax.plot_wireframe(y_track_nc_x, y_track_nc_y, np.zeros((1, len(y_track_nc))))
    plot_2d_dmp, = plt2d.plot(y_track_nc_x, y_track_nc_y, label='dmp')
    plt2d.scatter(y_track_nc_x, y_track_nc_y)
    # plot_paths.append(plot)
    # legend_key.append('dmp')
    # plt.show()
    # dmp_res = dmp.dt

    dmp_time_para = []
    dmp_dy_time_para = []
    for i in range(0, len(y_track_nc)):
        temp = list(y_track_nc[i])
        temp_vel = list(dy_track_nc[i])
        temp.append((i + 1) * dmp.dt)
        temp_vel.append((i + 1) * dmp.dt)
        dmp_dy_time_para.append(temp_vel)
        dmp_time_para.append(temp)

    dmp_time_para = np.array(dmp_time_para)
    dmp_dy_time_para = np.array(dmp_dy_time_para)

    rx, ry, rt, ucb = PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=[dmp_time_para], dmp_vel=dmp_dy_time_para)

    rx = np.array(rx)
    ry = np.array(ry)
    rt = np.array([rt])
    print("shape of rx is: ", rx.shape)

    print("rt is: ", rt)
    # plot = ax.plot_wireframe(rx, ry, rt)
    # plot_paths.append(plot)
    # legend_key.append('dijkstra')
    # plt.legend(plot_paths, legend_key, loc='lower right')
    plt2d.plot(rx, ry, label='final path')
    plt2d.legend()
    plt.show()


if __name__ == '__main__':
    x, y = get_trajectory("../csv/data.csv")
    print("length of demonstrated x trajectory is: ", len(x))

    print("[INFO]: Read x and y from the csv file")
    # plt.scatter(x, y)
    x = [0.1 * i for i in x]
    y = [0.1 * j for j in y]

    main(path_x=x, path_y=y)