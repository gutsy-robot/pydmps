"""
Probablistic Road Map (PRM) Planner
author: Atsushi Sakai (@Atsushi_twi)
"""

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

# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure()
plt2d = fig2.add_subplot(111)


class dmp_gen(rv_continuous):

    def __init__(self, dmp):
        self.dmp = dmp

    def _pdf(self, x, y):
        p = 1
        d = []
        for pt in self.dmp:
            distance = sqrt((x - pt[0] ** 2) + (y - pt[1]) ** 2)
            d.append(distance)
        d = np.array(d)
        ind = np.argmin(d)
        # x, y = self.dmp[ind][0], self.dmp[ind][1]
        dist = d[ind]

        return 0.005/(0.01 + dist)


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, t, cost, pind):
        self.x = x
        self.y = y
        self.t = t
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN
        inp: input data, single frame or multi frame
        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def PRM_planning(sx, sy, gx, gy, obstacles, dmp=None, dmp_vel=None):

    # obkdtree = KDTree(np.vstack((ox, oy)).T)

    sample_x, sample_y = sample_points(sx, sy, gx, gy, obstacles, dmp)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_roadmap(sample_x, sample_y, obstacles)

    rx, ry, rt = dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, obstacles, dmp, dmp_vel)

    return rx, ry, rt


def generate_roadmap(sample_x, sample_y, obstacles):

    print("min x index is: ", np.argmin(np.array(sample_x)))
    road_map = []
    nsample = len(sample_x)
    print("no of sample_x pts are: ", len(sample_x))
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        # print("length of index is: ", len(index))
        inds = index[0]
        # print("inds is: ", inds)
        # print("length of inds is: ", len(inds))
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]

            l = LineString([[ix, iy], [nx, ny]])
            for obstacle in obstacles:
                if not l.intersects(obstacle):
                    edge_id.append(inds[ii])

            if len(edge_id) >= N_KNN:
                break

        # print("edge id is: ", edge_id)
        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, obstacles, dmp=None,
                      dmp_vel=None):
    """
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    # vx_max = max(dmp_vel[:, 0])
    # vy_max = max(dmp_vel[:, 1])
    #
    # vx_min = min(dmp_vel[:, 0])
    # vy_min = min(dmp_vel[:, 1])

    # distance = sqrt(motion_y ** 2 + motion_x ** 2)
    #
    # del_tmax = distance / v_min
    # del_tmin = distance / v_max

    vel = []
    for v in dmp_vel:
        vel.append(math.sqrt(v[0] ** 2 + v[1] ** 2))

    vel = np.array(vel)

    v_max = max(vel)
    v_min = min(vel)

    print("v_max is: ", v_max)
    print("v_min is: ", v_min)
    print("length of roadmap is: ", len(road_map))
    print("length of sample_x is: ", len(sample_x))

    nstart = Node(sx, sy, 0.0, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if not openset:
            print("Cannot find path")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            ngoal.t = current.t
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            tmax = d/v_min
            tmin = d/v_max
            step = (tmax - tmin)/10

            for i in range(1, 11):
                delta_t = i * step
                dmp_cost = calculate_dmp_cost(current.x, current.y, current.t,  dx, dy, delta_t,
                                                 dmp, dmp_vel, obstacles)
                node = Node(sample_x[n_id], sample_y[n_id], current.t + delta_t,
                            current.cost + dmp_cost, c_id)

                if n_id in closedset:
                    continue
                # Otherwise if it is already in the open set
                if n_id in openset:
                    if openset[n_id].cost > node.cost:
                        openset[n_id].cost = node.cost
                        openset[n_id].pind = c_id
                else:
                    openset[n_id] = node

    # generate final course
    rx, ry, rt = [ngoal.x], [ngoal.y], [ngoal.t]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        rt.append(n.t)
        pind = n.pind

    return rx, ry, rt


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, obstacles, dmp=None):
    maxx = 150
    maxy = 150
    minx = 0
    miny = 0

    sample_x, sample_y = [], []

    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() - minx) * (maxx - minx)
        ty = (random.random() - miny) * (maxy - miny)

        point = Point((tx, ty))
        collision = False
        for obstacle in obstacles:
            if obstacle.contains(point):
                collision = True
                break
        if not collision:
            sample_x.append(tx)
            sample_y.append(ty)

    for pt in dmp:
        point = Point((pt[0], pt[1]))
        collision = False
        for obstacle in obstacles:
            if obstacle.contains(point):
                collision = True
                break

        if not collision:
            sample_x.append(pt[0])
            sample_y.append(pt[1])

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def calculate_dmp_cost(x, y, t, motion_x, motion_y, delta_t, dmp, dmp_vel, obstacles=None):

    d = []

    for pt in dmp:
        distance = sqrt((x + motion_x - pt[0]) ** 2 + (y + motion_y - pt[1]) ** 2 + (t + delta_t - pt[2]) ** 2)
        d.append(distance)

    d = np.array(d)

    min_index = np.argmin(d)

    # vx = motion_x / delta_t
    # vy = motion_y / delta_t
    #
    # d_vel = []
    #
    # for pt in dmp:
    #     # print("cost due to time is:  ", (t - pt[2]) ** 2)
    #     # print("cost due to x is:  ", (x - pt[0]) ** 2)
    #     distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2 + (t - pt[2]) ** 2)
    #     d_vel.append(distance)
    #
    # d_vel = np.array(d_vel)
    # min_index_vel = np.argmin(d_vel)

    cost = sqrt((x + motion_x - dmp[min_index][0]) ** 2 + (y + motion_y - dmp[min_index][1]) ** 2
                + (t + delta_t - dmp[min_index][2]) ** 2)
                # (vx - dmp_vel[min_index][0]) ** 2 + (vy - dmp_vel[min_index][1]) ** 2)
                # (t + delta_t - dmp[min_index][2]) ** 2)

    obstacle_cost = 0
    if obstacles is not None:
        for obstacle in obstacles:
            pol_ext = LinearRing(obstacle.exterior.coords)
            d = pol_ext.project(Point((x + motion_x), (y + motion_y)))
            p = pol_ext.interpolate(d)
            obst_potential_pt = list(p.coords)[0]
            dist = sqrt((y + motion_y - obst_potential_pt[1]) ** 2 +
                        (x + motion_x - obst_potential_pt[0]) ** 2)
            obstacle_cost += 100/((dist + 0.0000001) ** 2)

    # print("obstacle cost is: ", obstacle_cost)

    # cost += obstacle_cost

    # print("total cost is: ", cost)

    return cost


def main(path_x=None, path_y=None):
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 75.0  # [m]
    gy = 50.0  # [m]

    plot_paths = []
    legend_key = []

    z = np.zeros((1, len(path_x)))

    orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z, '--', linewidth=2)
    plt2d.plot(path_x, path_y, '--', linewidth=2)
    plot_paths.append(orig_path)
    legend_key.append('original')
    ax.scatter(path_x[0], path_y[0], z[0][0])
    plt2d.scatter(path_x[0], path_y[0])
    plt.annotate("st. original", (path_x[0], path_y[0]))
    ax.scatter(path_x[-1], path_y[-1], z[0][-1])
    plt2d.scatter(path_x[-1], path_y[-1])
    plt2d.annotate("f original.", (path_x[-1], path_y[-1]))

    ax.scatter(sx, sy, 0.0)
    plt2d.scatter(sx, sy)
    plt2d.annotate("new_st", (sx, sy))

    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    coords = [(50.0, 20.0), (50.0, 40.0), (60.0, 40.0), (60.0, 20.0)]
    poly1 = Polygon(coords)

    obstacles = [poly1]
    print("[INFO]: obstacles created")

    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    plt.plot(sx, sy, "xr")
    plt.plot(gx, gy, "xb")
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

    rx, ry, rt = PRM_planning(sx, sy, gx, gy, obstacles, dmp=dmp_time_para, dmp_vel=dmp_dy_time_para)

    rx = np.array(rx)
    ry = np.array(ry)
    rt = np.array([rt])
    print("shape of rx is: ", rx.shape)

    print("rt is: ", rt)
    plot = ax.plot_wireframe(rx, ry, rt)
    plot_paths.append(plot)
    legend_key.append('dijkstra')
    plt.legend(plot_paths, legend_key, loc='lower right')
    plt2d.plot(rx, ry)
    plt.show()

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    x, y = get_trajectory("../csv/data.csv")
    print("length of demonstrated x trajectory is: ", len(x))

    print("[INFO]: Read x and y from the csv file")
    # plt.scatter(x, y)
    x = [10 * i for i in x]
    y = [10 * j for j in y]

    main(path_x=x, path_y=y)