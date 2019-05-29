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
from mpl_toolkits.mplot3d import axes3d


N_SAMPLE = 500          # number of sample_points
N_KNN = 80              # number of edge from one sampled point
MAX_EDGE_LEN = 30.0     # [m] Maximum edge length

show_animation = True

fig, plt2d = plt.subplots()

fig2 = plt.figure(2)
plt_mean_reward = fig2.add_subplot(111)
fig2.suptitle('Mean Reward', fontsize=24)

fig3 = plt.figure(3)
plt_reward = fig3.add_subplot(111)
fig3.suptitle('Reward', fontsize=24)

fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')

fig5 = plt.figure(5)
plt_ucb = fig5.add_subplot(111)
fig5.suptitle('UCB Values', fontsize=24)

fig6 = plt.figure(6)
plt_connected = fig6.add_subplot(111)
fig6.suptitle('Number of disconnected components', fontsize=24)

dist_plotting = None


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


def plan(start, goal, guiding_paths, obstacles, v_max, v_min, num_points=3000,
         reward_weights={'connectivity': 0.01, 'increemental': 1.0}, guiding_path_weights=[1.0],
         ucb=None, uniform_only=False, normal_only=False):

    global dist_plotting

    print("plan_ucb called..")
    print("total number of nodes in the roadmap should be: ", num_points)

    conn_comp_arr = []

    dmp_time = guiding_paths[0]
    time_reso = dmp_time[1][2] - dmp_time[0][2]
    print("time reso is: ", time_reso)
    dmp_x_min = np.min(dmp_time[:, 0])
    dmp_y_min = np.min(dmp_time[:, 1])
    dmp_t_min = np.min(dmp_time[:, 2])

    dmp_x_max = np.max(dmp_time[:, 0])
    dmp_y_max = np.max(dmp_time[:, 1])
    dmp_t_max = np.max(dmp_time[:, 2])

    print("dmp_t_max is: ", dmp_t_max)
    avg_distance = 0.0
    for i in range(0, len(dmp_time) - 1):
        avg_distance += math.sqrt((dmp_time[i][0] - dmp_time[i+1][0]) ** 2 + (dmp_time[i][1] - dmp_time[i+1][1]) ** 2
                                  + (dmp_time[i][2] - dmp_time[i+1][2]) ** 2)

    avg_distance /= len(dmp_time)

    dist_plotting = 5 * avg_distance
    edge_resolution = avg_distance
    print("average 3D distance in the DMP is: ", avg_distance)

    print("min for uniform distribution is: ", (dmp_x_min, dmp_y_min, dmp_t_min))
    print("max for uniform distribution is: ", (dmp_x_max, dmp_y_max, dmp_t_max))

    tree = KDTree()
    tree.add(start.points[0][0], start.points[0][1])
    tree.add(goal.points[0][0], goal.points[0][1])

    print("start and goal added to the Kdtree..")

    vertices = [start, goal]
    roadmap = {0: [], 1: []}

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
            x, y, t = sample_uniform(dmp_x_min - 0.1, dmp_y_min - 0.1, 0,
                                    dmp_x_max + 0.1, dmp_y_max + 0.1, dmp_t_max + 0.1)

        elif arm == 1:

            # print("SAMPLED FROM DMP_NORMAL")
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

            # if arm == 0:
            #     if ucb is not None:
            #         plt2d.plot(x, y, marker="+", color='b', markersize=2.0)
            #
            # elif arm == 1:
            #     if ucb is not None:
            #         plt2d.plot(x, y, marker="+", color='r', markersize=2.0)

            # add the 2D node to the union find object
            edges = []
            node = Node([([x, y, t], [len(roadmap), 1/reward, None, arm])])
            # might be useful to vary the distance as a function of num_points for asym. optimality
            n_connected_components = uf.count()
            sampled_pt_id = len(roadmap) - num_goal_pts + 1

            neighbors = tree.neighbors((x, y, t), 5 * avg_distance)

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

        if ucb is not None:
            ucb.update(arm, reward)
            mean1.append(ucb.values[0])
            mean2.append(ucb.values[1])

    print("length of roadmap is: ", len(roadmap))

    if ucb is not None:
        plt_mean_reward.plot(mean1, 'bo', label='uniform')
        plt_mean_reward.plot(mean2, 'r+', label='dmp_normal')
        plt_mean_reward.legend()

        plt_reward.plot(increemental_reward, label='increemental')
        plt_reward.plot(connectivity_reward, label='connectivity')
        plt_reward.legend()

    # plt_connectivity_reward.plot(connectivity_reward)
    increemental_reward = np.array(increemental_reward)

    print("max in the increemental reward array is: ", np.max(increemental_reward))
    print("min in the increemental reward array is: ", np.min(increemental_reward))
    print("plt mean reward done..")

    if ucb is not None:
        plt_ucb.plot(ucb1, label='uniform')
        plt_ucb.plot(ucb2, label='dmp normal')
        plt_ucb.legend()

    if ucb is not None:
        plt_connected.plot(conn_comp_arr, label='ucb')

    elif uniform_only:
        plt_connected.plot(conn_comp_arr, label='uniform')

    elif normal_only:
        plt_connected.plot(conn_comp_arr, label='normal')

    return vertices, roadmap, ucb, edge_resolution


def dijkstra_planning(start, goal, road_map, vertices, guiding_paths=None, guiding_path_weights=None,
                      edge_resolution=None, use_discretised_cost=True, ucb_path=True):

    global dist_plotting
    # plotted_once = False

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
            if ucb_path:
                if current.points[0][1][3] == -1:
                    plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='g', markersize=8)

                elif current.points[0][1][3] == 0:
                    plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='b', markersize=8)

                elif current.points[0][1][3] == 1:
                    plt2d.plot(current.points[0][0][0], current.points[0][0][1], marker="x", color='r', markersize=8)

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

            node = vertices[n_id]

            if use_discretised_cost:
                edge_cost = calculate_discretised_edge_cost(node.points[0], vertices[c_id].points[0],
                                                            guiding_paths, guiding_path_weights,
                                                            edge_resolution)

                node.points[0][1][1] += edge_cost

            else:
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
        # if plotted_once is not True:
        #     circle = plt.Circle((n.points[0][0][0], n.points[0][0][1]), dist_plotting, linestyle='--',
        #                         linewidth=0.5, color='b', fill=False)
        #     # plt.Circle((n.points[0][0][0], n.points[0][0][1]), d)
        #     plt2d.add_patch(circle)
        #     plotted_once = True

        pind = n.points[0][1][2]

    return rx, ry, rt


def PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=None, dmp_vel=None, guiding_path_weights=[1.0],
                 reward_weights={'connectivity': 1.0, 'increemental': 0.1}, use_ucb=True, uniform_only=False,
                 normal_only=False):

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

    if use_ucb:
        ucb = UCB()
        ucb.initialize(2)

        vertices, roadmap, ucb_updated, edge_reso = plan(start, goal, guiding_paths, obstacles, v_max, v_min,
                                                         guiding_path_weights=guiding_path_weights,
                                                         ucb=ucb, reward_weights=reward_weights)
        rx, ry, rt = dijkstra_planning(start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                       guiding_paths=guiding_paths, edge_resolution=edge_reso)

    else:
        if uniform_only:
            vertices, roadmap, ucb_updated, edge_reso = plan(start, goal, guiding_paths, obstacles, v_max, v_min,
                                                             guiding_path_weights=guiding_path_weights,
                                                             uniform_only=True, reward_weights=reward_weights)
            rx, ry, rt = dijkstra_planning(start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                           guiding_paths=guiding_paths, edge_resolution=edge_reso, ucb_path=False)
        elif normal_only:
            vertices, roadmap, ucb_updated, edge_reso = plan(start, goal, guiding_paths, obstacles, v_max, v_min,
                                                             guiding_path_weights=guiding_path_weights,
                                                             normal_only=True, reward_weights=reward_weights)
            rx, ry, rt = dijkstra_planning(start, goal, roadmap, vertices, guiding_path_weights=guiding_path_weights,
                                           guiding_paths=guiding_paths, edge_resolution=edge_reso, ucb_path=False)

    return rx, ry, rt


def calculate_discretised_edge_cost(origin, destination, guiding_paths, guiding_path_weights, edge_resolution):

    edge_length = math.sqrt((origin[0][0] - destination[0][0]) ** 2 + (origin[0][1] - destination[0][1]) ** 2 +
                            (origin[0][2] - destination[0][2]) ** 2)

    if edge_length <= edge_resolution:
        return destination[1][1]

    else:

        guiding_path_index = np.argmax(np.array(guiding_path_weights))
        guiding_path = guiding_paths[guiding_path_index]

        mid_pt = [(origin[0][0] + destination[0][0]) / 2, (origin[0][1] + destination[0][1]) / 2,
                  (origin[0][2] + destination[0][2]) / 2]

        # print("mid_pt is: ", mid_pt)
        dist_array = []

        for pt in guiding_path:
            # print("pt from dmp is: ", pt)
            dist = math.sqrt((mid_pt[0] - pt[0]) ** 2 + (mid_pt[1] - pt[1]) ** 2 + (mid_pt[2] - pt[2]) ** 2)
            dist_array.append(dist)

        closest_pt_index = np.argmin(np.array(dist_array))

        cost = math.sqrt((mid_pt[0] - guiding_path[closest_pt_index][0]) ** 2 +
                         (mid_pt[1] - guiding_path[closest_pt_index][1]) ** 2)

        cost += destination[1][1]

        return cost


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

    orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z, '--', linewidth=2)
    ax.scatter(path_x[0], path_y[0], z[0][0])
    ax.scatter(path_x[-1], path_y[-1], z[0][-1])
    plt2d.plot(path_x, path_y, '--', linewidth=2, label='demonstration')

    plt2d.scatter(path_x[0], path_y[0])
    plt2d.annotate("st. original", (path_x[0], path_y[0]))
    plt2d.scatter(path_x[-1], path_y[-1])
    plt2d.annotate("f original.", (path_x[-1], path_y[-1]))

    ax.scatter(sx, sy, 0.0)
    plt2d.scatter(sx, sy)
    plt2d.annotate("new_st", (sx, sy))

    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    # coords = [(150.0, 120.0), (150.0, 140.0), (160.0, 140.0), (160.0, 120.0)]
    coords = [(50.0 * 0.01, 30.0 * 0.01), (50.0 * 0.01, 40.0 * 0.01), (60.0 * 0.01, 40.0 * 0.01),
              (60.0 * 0.01, 30.0 * 0.01)]
    poly1 = Polygon(coords)

    coords2 = [(25.0 * 0.01, 15.0 * 0.01), (25.0 * 0.01, 25.0 * 0.01), (35.0 * 0.01, 25.0 * 0.01),
               (35.0 * 0.01, 15.0 * 0.01)]
    poly2 = Polygon(coords2)

    obstacles = []
    # obstacles = []
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

    ax.scatter(y_track_nc_x, y_track_nc_y, y_track_nc_time)
    plot_2d_dmp, = plt2d.plot(y_track_nc_x, y_track_nc_y, label='dmp')
    plt2d.scatter(y_track_nc_x, y_track_nc_y)

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

    rx, ry, rt = PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=[dmp_time_para], dmp_vel=dmp_dy_time_para)

    rx_uniform, ry_uniform, rt_uniform = PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=[dmp_time_para],
                                                      dmp_vel=dmp_dy_time_para, use_ucb=False, uniform_only=True)

    rx_normal, ry_normal, rt_normal = PRM_planning(sx, sy, gx, gy, obstacles, guiding_paths=[dmp_time_para],
                                                   dmp_vel=dmp_dy_time_para, use_ucb=False, normal_only=True)
    rx = np.array(rx)
    ry = np.array(ry)
    rt = np.array([rt])
    print("shape of rx is: ", rx.shape)
    print("rt is: ", rt)

    rx_uniform = np.array(rx_uniform)
    ry_uniform = np.array(ry_uniform)
    rt_uniform = np.array([rt_uniform])

    print("rt uniform is: ", rt_uniform)

    rx_normal = np.array(rx_normal)
    ry_normal = np.array(ry_normal)
    rt_normal = np.array([rt_normal])

    print("rt uniform is: ", rt_normal)

    plot = ax.plot_wireframe(rx, ry, rt, color='k', label='ucb')
    # plot_paths.append(plot)
    # legend_key.append('ucb3D')

    plot = ax.plot_wireframe(rx_uniform, ry_uniform, rt_uniform, color='c', label='uniform')
    # plot_paths.append(plot)
    # legend_key.append('uniform3D')

    plot = ax.plot_wireframe(rx_normal, ry_normal, rt_normal, color='m', label='normal')
    # plot_paths.append(plot)
    # legend_key.append('normal3D')

    # plt.legend(plot_paths, legend_key, loc='lower right')
    ax.legend()
    plt2d.plot(rx, ry, color='k', label='ucb path')
    plt2d.plot(rx_uniform, ry_uniform, color='c', label='uniform path')
    plt2d.plot(rx_normal, ry_normal, color='m', label='normal path')
    plt2d.legend()
    plt_connected.legend()
    plt.show()


if __name__ == '__main__':
    x, y = get_trajectory("../csv/data.csv")
    print("length of demonstrated x trajectory is: ", len(x))

    print("[INFO]: Read x and y from the csv file")
    # plt.scatter(x, y)
    x = [0.1 * i for i in x]
    y = [0.1 * j for j in y]

    main(path_x=x, path_y=y)