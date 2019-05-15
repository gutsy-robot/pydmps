"""
Dijkstra grid based planning
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, mapping
from math import sqrt, ceil, floor
from utils import get_trajectory, check_collision, avoid_obstacles
from dmp_discrete import DMPs_discrete
import numpy as np
from mpl_toolkits.mplot3d import axes3d


show_animation = False
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig2 = plt.figure()
plt2d = fig2.add_subplot(111)


class Node(object):

    def __init__(self, x, y, t, cost, pind, dmp_closest_pt_index=None):
        self.x = x
        self.y = y
        self.t = t
        self.cost = cost
        self.pind = pind
        self.dmp_closest_pt_index = dmp_closest_pt_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.t) + "," + str(self.cost) + "," + str(self.pind)


def dijkstra_planning(sx, sy, gx, gy, obstacles, reso, reso_time=0.1, cost_type="default", dmp=None,
                      dmp_vel=None):
    """
    :param sx: start x coordinate
    :param sy: start y coordinate
    :param gx: goal x coordinate
    :param gy: goal y coordinate
    :param obstacles: list of shapely polygons
    :param reso: resolution for space
    :param reso_time: resolution for time
    :param cost_type: type of cost defining edge weights
    :param dmp: path given by a dmp
    :param dmp_vel: dmp_velocities
    :param dt: time resolution of the DMP
    :return: time parameterised path

    """
    
    # scale the dmp and its velocities according to the resolution of the grid.

    if cost_type == "dmp_traj":
        print("shape of dmp is: ", dmp.shape)
        print("shape of dmp velocity is: ", dmp_vel.shape)
        for i in range(0, len(dmp)):
            for j in range(0, len(dmp[i])):
                if j == 0 or j == 1:
                    dmp[i][j] = dmp[i][j] / reso
                    dmp_vel[i][j] = reso_time * dmp[i][j] / reso
                elif j == 2:
                    dmp[i][j] = dmp[i][j] / reso_time
                    dmp_vel[i][j] = dmp[i][j] / reso_time

    # print("dmp after scaling is: ", dmp)
    nstart = Node(round(sx / reso), round(sy / reso), 0.0, 0.0, -1)

    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, 0.0, -1)

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(obstacles, reso)

    motion = get_motion_model()
    print("motion is: ", motion)
    openset, closedset = dict(), dict()
    openset[calc_index(nstart, xw, minx, miny)] = nstart

    while 1:

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]
        # print("min dmp index of current active search node is: ", current.dmp_closest_pt_index)
        # print("t coordinate of min dmp cost index is: ", dmp[current.dmp_closest_pt_index])
        # print("coordinate of the node are: ", current.x, current.y, current.t)
        # show graph
        if show_animation:
            # print("doing search for a new point")
            ax.scatter(current.x * reso, current.y * reso, current.t * reso_time)
            plt2d.scatter(current.x * reso, current.y * reso)
            if len(closedset.keys()) % 10 == 0:
                plt.pause(0.0000000000001)

        if current.x == ngoal.x and current.y == ngoal.y:
            print("[INFO]: searched reached the goal")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            ngoal.t = current.t
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            if cost_type == "default":
                node = Node(current.x + motion[i][0], current.y + motion[i][1],
                            current.t + motion[i][2], current.cost + motion[i][3], c_id)
            elif cost_type == "dmp_traj":
                dmp_cost, min_index = calculate_dmp_cost(current.x, current.y, current.t,
                                              motion[i][0], motion[i][1], motion[i][2],
                                              dmp, dmp_vel, obstacles, reso=reso)

                node = Node(current.x + motion[i][0], current.y + motion[i][1],
                            current.t + motion[i][2], current.cost + dmp_cost,
                            c_id, min_index)
                # print("node is: ", node.t)
            n_id = calc_index(node, xw, minx, miny)
            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    rx, ry, rz = calc_final_path(ngoal, closedset, reso, reso_time)

    return rx, ry, rz


def calc_final_path(ngoal, closedset, reso, reso_time):
    # generate final course
    print("calculate final path called")
    rx, ry, rz = [ngoal.x * reso], [ngoal.y * reso], [ngoal.t * reso_time]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        rz.append(n.t * reso_time)
        pind = n.pind

    print("final path calculated..")
    return rx, ry, rz


def verify_node(node, obmap, minx, miny, maxx, maxy):

    # print("nodex and nodey are: ", (node.x, node.y))
    if obmap[node.x][node.y]:
        return False

    if node.x < minx:
        return False
    elif node.y < miny:
        return False
    elif node.x > maxx:
        return False
    elif node.y > maxy:
        return False

    return True


def calc_obstacle_map(obstacles, reso):

    minx = 0
    miny = 0
    maxx = 1000
    maxy = 1000
    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)

    # obstacle map generation
    obmap = [[False for i in range(xwidth)] for i in range(ywidth)]
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            #  print(x, y)
            point = Point((reso * x, reso * y))
            for obstacle in obstacles:
                if obstacle.contains(point):
                    obmap[ix][iy] = True
                    break

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    # return (node.y - ymin) * xwidth + (node.x - xmin) + node.t
    return node.x, node.y, node.t


def get_motion_model():

    motion = [[1, 0, 1, math.sqrt(2)],
              [-1, 0, 1, math.sqrt(2)],
              [0, 1, 1, math.sqrt(2)],
              [0, -1, 1, math.sqrt(2)],
              [0, 0, 1, 1]]
    return motion


def calculate_dmp_cost(x, y, t, motion_x, motion_y, delta_t, dmp, dmp_vel, obstacles=None, reso=1.0):
    """
    :param x: x coordinate of the point which we are moving to
    :param y: y coordinate of the point which we are moving to
    :param t: t coordinate of the point which we are moving to
    :param motion_x: distance to be moved along x
    :param motion_y: distance to be moved along y
    :param delta_t: movement along time axis
    :param dmp: reference dmp
    :param dmp_vel: reference dmp velocities
    :param obstacles: list of shapely polygons
    :param reso: grid resolution
    :return: cost of the given node

    """

    d = []

    for pt in dmp:
        # print("cost due to time is:  ", (t - pt[2]) ** 2)
        # print("cost due to x is:  ", (x - pt[0]) ** 2)
        distance = sqrt((x + motion_x - pt[0]) ** 2 + (y + motion_y - pt[1]) ** 2 + (t + delta_t - pt[2]) ** 2)
        d.append(distance)

    d = np.array(d)

    min_index = np.argmin(d)
    # print("min index is: ", min_index)

    vx = motion_x/ delta_t
    vy = motion_y/ delta_t

    d_vel = []

    for pt in dmp:
        # print("cost due to time is:  ", (t - pt[2]) ** 2)
        # print("cost due to x is:  ", (x - pt[0]) ** 2)
        distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2 + (t - pt[2]) ** 2)
        d_vel.append(distance)

    d_vel = np.array(d_vel)

    min_index_vel = np.argmin(d_vel)

    cost = sqrt((x + motion_x - dmp[min_index][0]) ** 2 + (y + motion_y - dmp[min_index][1]) ** 2 +
                (vx - dmp_vel[min_index_vel][0]) ** 2 + (vy - dmp_vel[min_index_vel][1]) ** 2)

    obstacle_cost = 0
    if obstacles is not None:
        for obstacle in obstacles:
            pol_ext = LinearRing(obstacle.exterior.coords)
            d = pol_ext.project(Point(reso * (x + motion_x), reso * (y + motion_y)))
            p = pol_ext.interpolate(d)
            obst_potential_pt = list(p.coords)[0]
            dist = sqrt((y + motion_y - obst_potential_pt[1]/reso) ** 2 +
                        (x + motion_x - obst_potential_pt[0]/reso) ** 2)
            obstacle_cost += 100/((dist + 0.0000001) ** 2)

    # print("obstacle cost is: ", obstacle_cost)

    cost += obstacle_cost

    # print("total cost is: ", cost)

    return cost, min_index


def main(sx=20.0, sy=10.0, gx=45.0,
         gy=60.0, grid_size=1.0, cost_type="default", path_x=None, path_y=None):

    plot_paths = []
    legend_key = []

    z = np.zeros((1, len(path_x)))
   
    orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z,  '--', linewidth=2)
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
    # ax.scatter(gx, gy, gt)
    # plt.show()
    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    coords = [(100.0, 20.0), (100.0, 40.0), (160.0, 40.0), (160.0, 20.0)]
    poly1 = Polygon(coords)

    obstacles = [poly1]
    print("[INFO]: obstacles created")

    # plot shapely polygons here:
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        print(np.array(x).shape)
        ax.plot_wireframe(np.array(x), np.array(y), np.zeros((1, len(x))))
        plt2d.plot(np.array(x), np.array(y))

    # plt.show()
    # plt.grid(True)
    # plt.axis("equal")
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
    plot_2d_dmp,  = plt2d.plot(y_track_nc_x, y_track_nc_y)
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
    
    rx, ry, rt = dijkstra_planning(sx, sy, gx, gy, obstacles, grid_size, cost_type="dmp_traj",
                                   dmp=dmp_time_para, dmp_vel=dmp_dy_time_para)

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


if __name__ == "__main__":
    x, y = get_trajectory("../csv/data.csv")
    print("length of demonstrated x trajectory is: ", len(x))
    # print("length of lists returned from get_traj is: ", le)

    # demo_x, demo_y = [], []
    #
    # for i in range(0, len(x)):
    #     if i % 20 == 0:
    #         demo_x.append(x[i])
    #         demo_y.append(y[i])

    print("[INFO]: Read x and y from the csv file")
    # plt.scatter(x, y)
    x = [10 * i for i in x]
    y = [10 * j for j in y]
    main(cost_type="dmp_traj", path_x=x, path_y=y)









