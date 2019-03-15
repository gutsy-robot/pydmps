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

show_animation = True


class Node(object):

    def __init__(self, x, y, cost, pind, time=0.0):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
        self.time = time

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


def dijkstra_planning(sx, sy, gx, gy, obstacles, reso, cost_type="default", dmp=None,
                      dmp_vel=None, dmp_res=None):
    """
    :param sx: start x coordinate
    :param sy: start y coordinate
    :param gx: goal x coordinate
    :param gy: goal y coordinate
    :param obstacles: list of shapely polygons
    :param reso: resolution of smaller grids
    :param cost_type: type of cost defining edge weights
    :param dmp: path given by a dmp
    :param dmp_vel: dmp_velocities
    :param dmp_res: time resolution of the DMP
    :return: time parameterised path

    """

    # scale the dmp and its velocities according to the resolution of the grid.

    if cost_type == "dmp_traj":
        print("shape of dmp is: ", dmp.shape)
        print("shape of dmp velocity is: ", dmp_vel.shape)
        for i in range(0, len(dmp)):
            for j in range(0, len(dmp[i])):
                dmp[i][j] = dmp[i][j] * reso
                dmp_vel[i][j] = dmp[i][j] * reso

    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)

    if cost_type == "dmp_traj":
        ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1, len(dmp))

    else:
        ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    # ox = [iox / reso for iox in ox]
    # oy = [ioy / reso for ioy in oy]

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(obstacles, reso)

    # print("obmap is: ", obmap)

    motion = get_motion_model()

    openset, closedset = dict(), dict()
    openset[calc_index(nstart, xw, minx, miny)] = nstart

    while 1:
        # print("start of while loop")
        # time.sleep(3.0)
        c_id = min(openset, key=lambda o: openset[o].cost)
        # print("cid assigned at the start of the loop: ", c_id)
        current = openset[c_id]
        # show graph
        if show_animation:
            plt.plot(current.x * reso, current.y * reso, "xc")
            # if len(closedset.keys()) % 10 == 0:
            #     plt.pause(0.0000000000001)

        if current.x == ngoal.x and current.y == ngoal.y:
            print("[INFO]: searched reached the goal")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            if cost_type == "default":
                node = Node(current.x + motion[i][0], current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            elif cost_type == "dmp_traj":
                dmp_cost, delta_t = calculate_dmp_cost(current.x, current.y,
                                                       motion[i][0], motion[i][1],
                                                       current.time, dmp, dmp_vel, obstacles)
                node = Node(current.x + motion[i][0], current.y + motion[i][1],
                            current.cost + dmp_cost,
                            c_id, current.time + delta_t)
                # print("current are: ", (current.x, current.y))
                # print("while instantiating the node, pind assigned is: ", c_id)

            n_id = calc_index(node, xw, minx, miny)
            #  print("nid is: ", n_id)

            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                # print("nid exists in openset")
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
                    # print("pind assigned is: ", c_id)
                    if cost_type == "dmp_traj":
                        openset[n_id].time = node.time
            else:
                openset[n_id] = node

    rx, ry = calc_final_path(ngoal, closedset, reso)

    return rx, ry


def calc_final_path(ngoal, closedset, reso):
    # generate final course
    print("calculate final path called")
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]
    pind = ngoal.pind
    while pind != -1:
        # print("pind is: ", pind)
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind

    print("final path calculated..")
    # print("rx is: ", rx)
    # print("ry is: ", ry)
    # plt.plot(rx, ry, 'r')
    # print("plot command executed")
    return rx, ry


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
    #  print("minx:", minx)
    #  print("miny:", miny)
    #  print("maxx:", maxx)
    #  print("maxy:", maxy)

    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)
    # print("xwidth:", xwidth)
    # print("ywidth:", ywidth)

    # obstacle map generation
    obmap = [[False for i in range(xwidth)] for i in range(ywidth)]
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            #  print(x, y)
            point = Point((reso * x, reso * y))
            for obstacle in obstacles:
                # print("obstacle is: ", obstacle)
                # for t in obstacle:
                #     for i in range(0, len(t)):
                #         t[i] = t[i] / reso

                if obstacle.contains(point):
                    obmap[ix][iy] = True
                    break

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)


def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion


def calculate_dmp_cost(x, y, motion_x, motion_y,  curr_time_index, dmp, dmp_vel, obstacles=None):
    """
    :param x: x coordinate of the point which we are moving to
    :param y: y coordinate of the point which we are moving to
    :param motion_x: distance to be moved along x
    :param motion_y: distance to be moved along y
    :param curr_time_index: current time in the trajectory(actual time = curr_time * dmp resolution)
    :param dmp: reference dmp
    :param dmp_vel: reference dmp velocities
    :param obstacles: list of shapely polygons
    :return: cost of the given node

    """

    # dmp_eff = dmp

    # dmp_eff = dmp[int(curr_time_index):]
    # pt, = plt.plot(x, y, 'bo')
    # plo, = plt.plot(dmp_eff[:, 0], dmp_eff[:, 1])
    # plt.pause(0.0001)
    # pt.remove()
    # plo.remove()

    d = []

    for pt in dmp:
        distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
        d.append(distance)

    d = np.array(d)

    time_index = np.argmin(d)

    delta_t_index = sqrt(motion_x ** 2 + motion_y ** 2)/sqrt(dmp_vel[time_index][0] ** 2 + dmp_vel[time_index][1] ** 2)

    # print("delta_t index is: ", delta_t_index)

    # print(curr_time_index + delta_t_index)
    dmp_0 = dmp[floor(time_index + delta_t_index)]

    if ceil(time_index + delta_t_index) < len(dmp):
        dmp_1 = dmp[ceil(time_index + delta_t_index)]
        dmp_next = dmp_0 + (dmp_1 - dmp_0) * delta_t_index
    # dmp_next = dmp[ceil(curr_time_index + delta_t_index)]

    else:
        dmp_next = dmp_0

    dmp_x = dmp_next[0]
    dmp_y = dmp_next[1]

    obstacle_cost = 0
    if obstacles is not None:
        for obstacle in obstacles:
            pol_ext = LinearRing(obstacle.exterior.coords)
            d = pol_ext.project(Point(x + motion_x, y + motion_y))
            p = pol_ext.interpolate(d)
            obst_potential_pt = list(p.coords)[0]
            dist = sqrt((y + motion_y - obst_potential_pt[1]) ** 2 + (x + motion_x - obst_potential_pt[0]) ** 2)
            obstacle_cost += 10/((dist + 0.0000001) ** 2)

    print("obstacle cost is: ", obstacle_cost)
    cost = sqrt((y + motion_y - dmp_y) ** 2 + (x + motion_x - dmp_x) ** 2) + obstacle_cost
    print("total cost is: ", cost)

           # + sqrt((y - dmp[time_index][1]) ** 2 + (x - dmp[time_index][0]) ** 2)

    # print("cost is: ", cost)
    return cost, delta_t_index


def main(sx=20.0, sy=10.0, gx=100.0, gy=100.0,
         grid_size=1.0, cost_type="default", path_x=None, path_y=None, n_bfs=[100]):

    print(__file__ + " start!!")

    # start and goal position
    # sx = 10.0  # [m]
    # sy = 10.0  # [m]
    # gx = 50.0  # [m]
    # gy = 50.0  # [m]
    # grid_size = 1.0  # [m]

    legend_key = []
    plot_paths = []
    if cost_type == "dmp_traj":
        orig_path, = plt.plot(path_x, path_y, '--', linewidth=2)

        plot_paths.append(orig_path)
        legend_key.append('original')
        plt.plot(path_x[0], path_y[0], 'bo')
        plt.annotate("st. original", (path_x[0], path_y[0]))
        plt.plot(path_x[-1], path_y[-1], 'bo')
        plt.annotate("f original.", (path_x[-1], path_y[-1]))

    plt.plot(sx, sy, 'bo')
    plt.annotate("new_st", (sx, sy))
    plt.plot(gx, gy, 'bo')
    plt.annotate("new_fin", (gx, gy))

    coords = [(50.0, 20.0), (50.0, 40.0), (60.0, 40.0), (60.0, 20.0)]
    poly1 = Polygon(coords)

    # coords2 = [(6.0, 6.5), (6.0, 8.0), (8.0, 8.0), (8.0, 6.5)]
    # poly2 = Polygon(coords2)

    obstacles = [poly1]
    print("[INFO]: obstacles created")

    if show_animation:
        # plt.plot(ox, oy, ".k")

        # plot shapely polygons here:
        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        # plt.plot(sx, sy, "xr")
        # plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        print("[INFO]: Plotted obstacles and start and end pts..")

    if cost_type == "dmp_traj":
        for ii, bfs in enumerate(n_bfs):
            dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs, dt=0.01)

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

            plot, = plt.plot(y_track_nc[:, 0], y_track_nc[:, 1])
            plot_paths.append(plot)
            legend_key.append('no_avoidance')

            y_track, dy_track, ddy_track, s = dmp.rollout(external_force=avoid_obstacles,
                                                          obstacles=obstacles, gamma=0.1)

            plot, = plt.plot(y_track[:, 0], y_track[:, 1])
            plot_paths.append(plot)
            legend_key.append('pot. field')


            #
            # for i in range(0, len(y_track_nc)):
            #     plt.plot(y_track_nc[i][0], y_track_nc[i][1], 'bo')

            # path_points = []
            #
            # for point in y_track_nc:
            #     path_points.append(Point(tuple(point)))
            #
            # path_intersect = check_collision(path_points, obstacles)

            dmp_res = dmp.dt

            # rx, ry = dijkstra_planning(sx, sy, gx, gy, obstacles, grid_size)
            print("calling dijkstra's planning..")
            rx, ry = dijkstra_planning(sx, sy, gx, gy, obstacles, grid_size, cost_type="dmp_traj",
                                       dmp=y_track_nc, dmp_vel=dy_track_nc, dmp_res=dmp_res)

            rx = np.array(rx)
            print("shape of rx is: ", rx.shape)

    elif cost_type == "default":
        rx, ry = dijkstra_planning(sx, sy, gx, gy, obstacles, grid_size)

    if show_animation:
        plot, = plt.plot(list(rx), ry, "-r")
        plot_paths.append(plot)
        legend_key.append('dijkstra')
        plt.legend(plot_paths, legend_key, loc='lower right')

        plt.show()


x, y = get_trajectory("../csv/data.csv")
# x = [i * 10.0 for i in x]
# y = [j * 10.0 for j in y]
print("[INFO]: Read x and y from the csv file")

main(cost_type="dmp_traj", path_x=x, path_y=y)
# main()
