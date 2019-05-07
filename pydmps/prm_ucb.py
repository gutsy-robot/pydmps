import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from shapely.geometry import Point, mapping
from math import sqrt, ceil, floor
from utils import get_trajectory, check_collision, avoid_obstacles
from dmp_discrete import DMPs_discrete
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import rv_continuous
# from prm import Node
from kdtree import Node, KDTree


# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 80  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure(2)
plt2d = fig2.add_subplot(111)

fig3 = plt.figure(3)
plt_ucb = fig3.add_subplot(111)


def sample_dmp_normal(x_dmp, y_dmp, t_dmp):
    mean = [x_dmp, y_dmp, t_dmp]
    cov = [[10, 0, 0], [0, 10, 0], [0, 0, 0.1]]
    x, y, t = np.random.multivariate_normal(mean, cov, 1).T

    return x, y, t


def sample_unform(minx, miny, mint, maxx, maxy, maxt):
    x = (random.random() - minx) * (maxx - minx)
    y = (random.random() - miny) * (maxy - miny)
    t = (random.random() - mint) * (maxt - mint)
    # if math.sqrt((x - 10.0) ** 2 + (y - 10.0) ** 2 + t **2) < 9.0:
    #     print("point sampled should be in the roadmap of t[0]")

    return x, y, t


def ind_max(x):
    m = max(x)
    return x.index(m)


class UCB:
    def __init__(self, counts=None, values=None):
        self.counts = counts
        self.values = values
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]

        return

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return ind_max(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


def get_reward(x, y, t, dmp, obstacles=None):

    d = []

    for pt in dmp:
        distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2 + (t - pt[2]) ** 2)
        d.append(distance)

    d = np.array(d)

    min_index = np.argmin(d)
    cost = sqrt((x - dmp[min_index][0]) ** 2 + (y - dmp[min_index][1]) ** 2
                + (t - dmp[min_index][2]) ** 2)

    obstacle_cost = 0
    if obstacles is not None:
        for obstacle in obstacles:
            point = Point((pt[0], pt[1]))
            if obstacle.contains(point):
                cost = 0
                return cost
            else:
                pol_ext = LinearRing(obstacle.exterior.coords)
                d = pol_ext.project(point)
                p = pol_ext.interpolate(d)
                obst_potential_pt = list(p.coords)[0]
                dist = sqrt((y - obst_potential_pt[1]) ** 2 +
                            (x - obst_potential_pt[0]) ** 2)
                obstacle_cost += 100 / ((dist + 0.0000001) ** 2)

    # cost += obstacle_cost
    # print("reward returned is: ", (1/cost))
    return 1/cost


def plan_ucb(start, goal, dmp_time, obstacles, v_max, v_min, num_points=3000):

    print("plan_ucb called..")

    time_reso = dmp_time[1][2] - dmp_time[0][2]
    dmp_x_min = np.min(dmp_time[:, 0])
    dmp_y_min = np.min(dmp_time[:, 1])
    dmp_t_min = np.min(dmp_time[:, 2])

    dmp_x_max = np.max(dmp_time[:, 0])
    dmp_y_max = np.max(dmp_time[:, 1])
    dmp_t_max = np.max(dmp_time[:, 2])

    print("min for uniform distribution is: ", (dmp_x_min, dmp_y_min, dmp_t_min))
    print("max for uniform distribution is: ", (dmp_x_max, dmp_y_max, dmp_t_max))

    ucb = UCB()
    n_distributions = 2
    # n_distributions = len(dmp_time) + 1
    ucb.initialize(n_distributions)
    ucb1 = [0.0]
    ucb2 = [0.0]
    # sample_x = [start.x, goal.x]
    # sample_y = [start.y, goal.y]
    # sample_t = [start.t, goal.t]
    tree = KDTree()
    tree.add(start.points[0][0], start.points[0][1])
    tree.add(goal.points[0][0], goal.points[0][1])

    print("start and goal added to the Kdtree..")

    vertices = [start, goal]
    roadmap = {0: [], 1: []}
    # num_points_roadmap = 2

    for i in range(1, 20):
        # sample_x.append(goal.x)
        # sample_y.append(goal.y)
        # print("goal.points[1] is: ", goal.points[1])
        t = i * time_reso + goal.points[0][0][2]
        # sample_t.append(t)
        # n = Node(goal.x, goal.y, t, i * time_reso, -1)
        n = Node([([goal.points[0][0][0], goal.points[0][0][1], t],
                  [len(vertices), i * time_reso, -1])])
        # print("point is: ", n.points[0][0])
        # print("data is: ", n.points[0][1])
        tree.add(n.points[0][0], n.points[0][1])
        # print("added node with cid: ", len(vertices))
        vertices.append(n)
        roadmap[len(roadmap)] = []

    # print("multiple goal nodes added to the roadmap..")
    while len(vertices) < num_points:
        arm = ucb.select_arm()
        print("values are: ", ucb.values)
        # print("arm selected is ", arm)
        if arm == 0:
            x, y, t = sample_unform(0, 0, 0,
                                    dmp_x_max + 5, dmp_y_max + 5, dmp_t_max + 0.1)

        else:
            i = random.randint(0, len(dmp_time) - 1)
            x, y, t = sample_dmp_normal(dmp_time[i][0], dmp_time[i][1], dmp_time[i][2])

        # else:
        #     x, y, t = sample_unform(0, 0, 0,
        #                             dmp_x_max + 2, dmp_y_max + 2, dmp_t_max + 0.5)

        reward = get_reward(x, y, t, dmp_time, obstacles)
        ucb.update(arm, reward)
        ucb1.append(ucb.values[0] * 10000)
        ucb2.append(ucb.values[1] * 10000)

        if reward > 0:
            edges = []
            node = Node([([x, y, t], [len(roadmap), 1/reward, None])])
            # node = Node(x, y, t, 1/reward)
            # print("cost of added node is: ", (1/reward))
            ax.scatter(x, y, t)
            # might be useful to vary the distance as a function of num_points for asym. optimality

            neighbors = tree.neighbors((x, y, t), 10)
            # print("neighbors returned within 5 units radius")
            # print("neighbors are: ", neighbors)
            for n in neighbors:
                # print("n.points is: ", n.points)
                x_out = n[0][0]
                y_out = n[0][1]
                t_out = n[0][2]

                dist_2d = math.sqrt((x - x_out) ** 2 + (y - y_out) ** 2)
                if t_out > t:
                    vel = dist_2d / (t_out - t)
                    if v_min <= vel <= v_max:
                        l = LineString([[x, y], [x_out, y_out]])
                        for obstacle in obstacles:
                            if not l.intersects(obstacle):
                                edges.append(n[1][0])

                elif t_out < t:
                    vel = dist_2d / (t - t_out)
                    if v_min <= vel < v_max:
                        l = LineString([[x, y], [x_out, y_out]])
                        for obstacle in obstacles:
                            if not l.intersects(obstacle):
                                roadmap[n[1][0]].append(len(roadmap))

            # for i in range(0, len(sample_x)):
            #     x_out = sample_x[i]
            #     y_out = sample_y[i]
            #     t_out = sample_t[i]
            #
            #     dist_3d = math.sqrt((x - x_out) ** 2 + (y - y_out) ** 2 + (t - t_out) ** 2)
            #     dist_2d = math.sqrt((x - x_out) ** 2 + (y - y_out) ** 2)
            #     # if i == 0:
            #     #     print("for i equal to zero distance 3D is: ", dist_3d)
            #     #     print("distance 2D is: ", dist_2d)
            #
            #     if dist_3d < 10:
            #         if t_out > t:
            #             vel = dist_2d / (t_out - t)
            #             if v_min <= vel < v_max:
            #                 l = LineString([[x, y], [x_out, y_out]])
            #                 for obstacle in obstacles:
            #                     if not l.intersects(obstacle):
            #                         edges.append(i)
            #             #             if i == 0:
            #             #                 print("something added for roadmap[0]")
            #             # else:
            #             #     if i == 0:
            #             #         print("velocity not in range for i equal to zero")

                    # elif t_out < t:
                    #     vel = dist_2d / (t - t_out)
                    #     if v_min <= vel < v_max:
                    #         l = LineString([[x, y], [x_out, y_out]])
                    #         for obstacle in obstacles:
                    #             if not l.intersects(obstacle):
                    #                 roadmap[i].append(len(sample_x))

            tree.add(node.points[0][0], node.points[0][1])
            roadmap[len(vertices)] = edges
            vertices.append(node)
            # print("node added to roadmap and vertices array")

            # # this block not required
            # sample_x.append(x)
            # sample_y.append(y)
            # sample_t.append(t)

    # print("length of sample_x is: ", len(sample_x))
    print("length of roadmap is: ", len(roadmap))

    print("roadmap[1] is: ", roadmap[1])
    # print("one of the nodes are: ", vertices[roadmap[1][0]].points[0][0])
    # return sample_x, sample_y, sample_t, vertices, roadmap
    plt_ucb.plot(ucb1, 'bo')
    plt_ucb.plot(ucb2, 'r+')
    # print("ucb1 is: ", ucb1)
    # print("ucb2 is: ", ucb2)
    # plt_ucb.show()
    return vertices, roadmap


def dijkstra_planning(start, goal, road_map, vertices):

    # nstart = Node(sx, sy, 0.0, 0.0, -1)
    # ngoal = Node(gx, gy, 0.0, 0.0, -1)

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

        # have to change the way cost is accessed
        # c_id = min(openset, key=lambda o: openset[o].cost)
        # print("length of openset is: ", len(openset))
        min = None
        c_id = None
        for k, v in openset.items():
            cost = v.points[0][1][1]
            if min is None:
                min = cost
                c_id = v.points[0][1][0]
                # print("cid is: ", c_id)

            else:
                if cost < min:
                    min = cost
                    c_id = v.points[0][1][0]
                    # print("cid is: ", c_id)

        # c_id = openset[openset.index(min([x.points[1][1] for key, x in openset.items()]))][1][0]
        # print("doing operations for c_id: ", c_id)
        # c_id = min(openset, key=lambda o: openset[o][1][1])
        if c_id in range(0, 21):
            print('yes..')
        current = openset[c_id]

        # show graph
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt2d.plot(current.points[0][0][0], current.points[0][0][1], "xg")
            ax.scatter(current.points[0][0][0], current.points[0][0][1], current.points[0][0][2])
            plt.pause(0.001)

        if c_id in range(1, 21):
            print("goal is found!")
            print("c_id is: ", c_id)
            goal.points[0][1][2] = current.points[0][1][2]
            goal.points[0][1][1] = current.points[0][1][1]
            goal.points[0][0][2] = current.points[0][0][2]
            # goal.pind = current.pind
            # goal.cost = current.cost
            # print("cost is: ", current.cost)
            # goal.t = current.t
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
            # if n_id == 1:
            #     print("goal at time o reached")
            #     print("cost of the current node is: ", current.cost)
            # if n_id in range(2, 19):
            #     print("goal reached at non-zero time..")
            #     print("cost of the current node is: ", current.cost)

            node = vertices[n_id]
            # node = vertices[n_id]
            # print("cost of node before connection: ", node.cost)
            # node.cost += current.cost
            # print(vertices[c_id].points[0][1][1])
            # temp = node.points[1][1]
            node.points[0][1][1] += vertices[c_id].points[0][1][1]
            # print("cost of node after connection: ", node.cost)
            # node.pind = c_id
            node.points[0][1][2] = c_id

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].points[0][1][1] > node.points[0][1][1]:
                    openset[n_id].points[0][1][1] = node.points[0][1][1]
                    openset[n_id].points[0][1][2] = c_id
                # if openset[n_id].cost > node.cost:
                #     openset[n_id].cost = node.cost
                #     openset[n_id].pind = c_id

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

    # rx, ry, rt = [goal.x], [goal.y], [goal.t]
    # pind = goal.pind
    # while pind != -1:
    #     n = closedset[pind]
    #     rx.append(n.x)
    #     ry.append(n.y)
    #     rt.append(n.t)
    #     pind = n.pind

    return rx, ry, rt


def PRM_planning(sx, sy, gx, gy, obstacles, dmp_time=None, dmp_vel=None):

    # declare node as ((x, y, t), (id, cost, pind))
    start = Node([([sx, sy, 0], [0, 0, -1])])
    # start = Node(sx, sy, 0, 0, -1)

    print("time at goal is: ", dmp_time[-1][2])
    goal = Node([([gx, gy, dmp_time[-1][2]], [1, 0, -1])])
    # goal = Node(gx, gy, dmp_time[-1][2], 0, -1)

    print("start and goal nodes declared..")
    vel = []
    for v in dmp_vel:
        vel.append(math.sqrt(v[0] ** 2 + v[1] ** 2))

    vel = np.array(vel)

    v_max = max(vel)
    v_min = min(vel)

    print("vmax is: ", v_max)
    print("vmin is: ", v_min)

    # sample_x, sample_y, sample_t, vertices, roadmap = plan_ucb(start, goal, dmp_time, obstacles, v_max, v_min)

    vertices, roadmap = plan_ucb(start, goal, dmp_time, obstacles, v_max, v_min)
    rx, ry, rt = dijkstra_planning(start, goal, roadmap, vertices)

    return rx, ry, rt


def main(path_x=None, path_y=None):
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]

    # sx = 100.0
    # sy = 100.0
    gx = 75.0  # [m]
    gy = 70.0  # [m]

    plot_paths = []
    legend_key = []

    z = np.zeros((1, len(path_x)))

    orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z, '--', linewidth=2)
    plt2d.plot(path_x, path_y, '--', linewidth=2)
    plot_paths.append(orig_path)
    legend_key.append('original')
    ax.scatter(path_x[0], path_y[0], z[0][0])
    plt2d.scatter(path_x[0], path_y[0])
    plt2d.annotate("st. original", (path_x[0], path_y[0]))
    ax.scatter(path_x[-1], path_y[-1], z[0][-1])
    plt2d.scatter(path_x[-1], path_y[-1])
    plt2d.annotate("f original.", (path_x[-1], path_y[-1]))

    ax.scatter(sx, sy, 0.0)
    plt2d.scatter(sx, sy)
    plt2d.annotate("new_st", (sx, sy))

    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    # coords = [(150.0, 120.0), (150.0, 140.0), (160.0, 140.0), (160.0, 120.0)]
    coords = [(150.0, 120.0), (150.0, 140.0), (160.0, 140.0), (160.0, 120.0)]
    poly1 = Polygon(coords)

    obstacles = [poly1]
    print("[INFO]: obstacles created")

    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        plt2d.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    plt2d.plot(sx, sy, "xr")
    plt2d.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")
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
    # print("y_track_nc is: ", y_track_nc)

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
    # ax.plot_wireframe(y_track_nc_x, y_track_nc_y, y_track_nc_time)
    plot = ax.plot_wireframe(y_track_nc_x, y_track_nc_y, np.zeros((1, len(y_track_nc))))
    plot_2d_dmp, = plt2d.plot(y_track_nc_x, y_track_nc_y)
    plt2d.scatter(y_track_nc_x, y_track_nc_y)
    plot_paths.append(plot)
    legend_key.append('dmp')
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

    # print("dmp time para is: ", dmp_time_para)
    # print("dmp velocity time para is: ", dmp_dy_time_para)

    rx, ry, rt = PRM_planning(sx, sy, gx, gy, obstacles, dmp_time=dmp_time_para, dmp_vel=dmp_dy_time_para)

    rx = np.array(rx)
    ry = np.array(ry)
    rt = np.array([rt])
    print("shape of rx is: ", rx.shape)

    print("rt is: ", rt)
    plot = ax.plot_wireframe(rx, ry, rt)
    plot_paths.append(plot)
    legend_key.append('dijkstra')
    # plt.legend(plot_paths, legend_key, loc='lower right')
    plt2d.plot(rx, ry)
    plt.show()

    # if show_animation:
    #     plt.plot(rx, ry, "-r")
    #     plt.show()


if __name__ == '__main__':
    x, y = get_trajectory("../csv/data.csv")
    print("length of demonstrated x trajectory is: ", len(x))

    print("[INFO]: Read x and y from the csv file")
    # plt.scatter(x, y)
    x = [10 * i for i in x]
    y = [10 * j for j in y]

    main(path_x=x, path_y=y)