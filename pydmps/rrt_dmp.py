#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, mapping
from math import sqrt, ceil, floor
from utils import get_trajectory, check_collision, avoid_obstacles
from dmp_discrete import DMPs_discrete
import numpy as np
from grid_search import Node
from multiprocessing import Process
import time


class RrtDmps(object):
    def __init__(self, start_x=20.0, start_y=10.0, goal_x=60.0, goal_y=60.0, obstacles=None, grid_size=1.0,
                 cost_type="dmp_traj", demo_x=None, demo_y=None, n_basis=[100], animation=True):

        print("RRTDMP object initialising..")
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.cost_type = cost_type
        self.demo_x = demo_x
        self.demo_y = demo_y
        self.n_basis = n_basis

        self.forward_graph = []
        self.reverse_graph = []
        self.dmp = None
        self.dmp_dy = None
        self.dmp_ddy = None
        self.show_animation = animation
        self.obmap = None
        self.legend = []
        self.plots = []

        self.motion_model = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [-1, -1, math.sqrt(2)],
                             [-1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

        self.minx = 0
        self.miny = 0
        self.maxx = 1000
        self.maxy = 1000
        self.xwidth = round(self.maxx - self.minx)
        self.ywidth = round(self.maxy - self.miny)
        self.found_path = False
        self.r1_x = None
        self.r1_y = None

        self.r2_x = None
        self.r2_y = None
        print("RRT DMP object initialised..")

    def plot(self):

        print("plot method called..")
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

        plt.grid(True)
        plt.axis("equal")

        plt.plot(self.start_x, self.start_y, 'bo')
        plt.annotate("new_st", (self.start_x, self.start_y))
        plt.plot(self.goal_x, self.goal_y, 'bo')
        plt.annotate("new_fin", (self.goal_x, self.goal_y))

        if self.demo_x is not None and self.demo_y is not None:
            orig_path, = plt.plot(self.demo_x, self.demo_y, '--', linewidth=2)

            self.plots.append(orig_path)
            self.legend.append('original')
            plt.plot(self.demo_x[0], self.demo_y[0], 'bo')
            plt.annotate("st. original", (self.demo_x[0], self.demo_y[0]))
            plt.plot(self.demo_x[-1], self.demo_y[-1], 'bo')
            plt.annotate("f original.", (self.demo_x[-1], self.demo_y[-1]))

        if self.dmp is not None:
            plot, = plt.plot(self.dmp[:, 0], self.dmp[:, 1])
            self.plots.append(plot)
            self.legend.append('no_avoidance')

    def calc_obstacle_map(self):
        print("calculate obstacle map called..")
        # obstacle map generation
        self.obmap = [[False for i in range(self.xwidth)] for i in range(self.ywidth)]
        for ix in range(self.xwidth):
            x = ix + self.minx
            for iy in range(self.ywidth):
                y = iy + self.miny
                point = Point((self.grid_size * x, self.grid_size * y))
                for obstacle in self.obstacles:
                    if obstacle.contains(point):
                        self.obmap[ix][iy] = True
                        break
        print("obmap calculated..")

    def calc_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def calculate_dmp_cost(self, x, y, motion_x, motion_y, dmp, dmp_vel):
        """
        :param x: x coordinate of the point which we are moving to
        :param y: y coordinate of the point which we are moving to
        :param motion_x: distance to be moved along x
        :param motion_y: distance to be moved along y
        :param dmp: reference dmp
        :param dmp_vel: reference dmp velocities
        :return: cost of the given node

        """

        d = []

        for pt in dmp:
            distance = sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
            d.append(distance)

        d = np.array(d)

        time_index = np.argmin(d)

        delta_t_index = sqrt(motion_x ** 2 + motion_y ** 2) / sqrt(dmp_vel[time_index][0] ** 2
                                                                   + dmp_vel[time_index][1] ** 2)

        dmp_0 = dmp[floor(time_index + delta_t_index)]

        if ceil(time_index + delta_t_index) < len(dmp):
            dmp_1 = dmp[ceil(time_index + delta_t_index)]
            dmp_next = dmp_0 + (dmp_1 - dmp_0) * delta_t_index

        else:
            dmp_next = dmp_0

        dmp_x = dmp_next[0]
        dmp_y = dmp_next[1]

        obstacle_cost = 0
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                pol_ext = LinearRing(obstacle.exterior.coords)
                d = pol_ext.project(Point(x + motion_x, y + motion_y))
                p = pol_ext.interpolate(d)
                obst_potential_pt = list(p.coords)[0]
                dist = sqrt((y + motion_y - obst_potential_pt[1]) ** 2 + (x + motion_x - obst_potential_pt[0]) ** 2)
                obstacle_cost += 100 / ((dist + 0.0000001) ** 2)

        print("obstacle cost is: ", obstacle_cost)
        cost = sqrt((y + motion_y - dmp_y) ** 2 + (x + motion_x - dmp_x) ** 2) + obstacle_cost - sqrt((y - dmp[
            time_index][1]) ** 2 + (x - dmp[time_index][0]) ** 2)

        print("total cost is: ", cost)

        return cost

    def verify_node(self, node):

        if self.obmap[node.x][node.y]:
            return False

        if node.x < self.minx:
            return False
        elif node.y < self.miny:
            return False
        elif node.x > self.maxx:
            return False
        elif node.y > self.maxy:
            return False

        return True

    def reverse_dmp(self):
        print("reverse dmp called..")
        dmp_rev = self.dmp
        np.flip(dmp_rev, axis=0)
        dmp_rev_dy = self.dmp_dy
        np.flip(dmp_rev_dy, axis=0)
        for i in range(0, len(dmp_rev_dy)):
            for j in range(0, len(dmp_rev_dy[i])):
                dmp_rev_dy[i][j] *= -1
        return dmp_rev, dmp_rev_dy

    def calc_final_path(self, ngoal, graph_lookup="reverse"):
        # generate final course

        print("calculate final path called")
        rx, ry = [ngoal.x * self.grid_size], [ngoal.y * self.grid_size]
        pind = ngoal.pind
        while pind != -1:
            if graph_lookup == "reverse":
                n = self.forward_graph[pind]
            elif graph_lookup == "forward":
                n = self.reverse_graph[pind]

            rx.append(n.x * self.grid_size)
            ry.append(n.y * self.grid_size)
            pind = n.pind

        print("final path calculated..")

        return rx, ry

    def find_path(self, sx, sy, dmp=None, dmp_dy=None, graph_lookup="reverse"):

        print("find path called..")
        if self.cost_type == "dmp_traj":
            print("shape of dmp is: ", self.dmp.shape)
            print("shape of dmp velocity is: ", self.dmp_dy.shape)
            for i in range(0, len(dmp)):
                for j in range(0, len(dmp[i])):
                    dmp[i][j] = dmp[i][j] * self.grid_size
                    dmp_dy[i][j] = dmp_dy[i][j] * self.grid_size

        nstart = Node(round(sx / self.grid_size), round(sy / self.grid_size), 0.0, -1)

        # ngoal = Node(round(gx / self.grid_size), round(gy / self.grid_size), 0.0, -1)

        ngoal = None
        motion = self.motion_model

        if graph_lookup == "reverse":
            openset, self.forward_graph= dict(), dict()

        elif graph_lookup == "forward":
            openset, self.reverse_graph = dict(), dict()

        openset[self.calc_index(nstart)] = nstart

        while 1:
            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]
            # show graph
            if self.show_animation:
                plt.plot(current.x * self.grid_size, current.y * self.grid_size, "xc")
                if graph_lookup == "reverse":
                    if len(self.forward_graph.keys()) % 10 == 0:
                        plt.pause(0.0000000000001)
                elif graph_lookup == "forward":
                    if len(self.reverse_graph.keys()) % 10 == 0:
                        plt.pause(0.0000000000001)

            if graph_lookup == "reverse":
                pts = [(node.x, node.y) for node in self.reverse_graph]
                if (current.x, current.y) in pts:
                    print("[INFO]: searched reached the goal")
                    ngoal = current
                    break

            elif graph_lookup == "front":
                pts = [(node.x, node.y) for node in self.forward_graph]
                if (current.x, current.y) in pts:
                    ngoal = current
                    break

            # Remove the item from the open set
            del openset[c_id]
            # Add it to the closed set
            if graph_lookup == "reverse":
                self.forward_graph[c_id] = current

            elif graph_lookup == "forward":
                self.reverse_graph[c_id] = current

            # expand search grid based on motion model
            for i, _ in enumerate(motion):
                if self.cost_type == "default":
                    node = Node(current.x + motion[i][0], current.y + motion[i][1],
                                current.cost + motion[i][2], c_id)
                elif self.cost_type == "dmp_traj":
                    dmp_cost = self.calculate_dmp_cost(current.x, current.y, motion[i][0],
                                                       motion[i][1], dmp, dmp_dy)
                    node = Node(current.x + motion[i][0], current.y + motion[i][1],
                                current.cost + dmp_cost,
                                c_id)

                n_id = self.calc_index(node)

                if not self.verify_node(node):
                    continue

                if graph_lookup == "reverse":
                    if n_id in self.forward_graph:
                        continue
                elif graph_lookup == "forward":
                    if n_id in self.reverse_graph:
                        continue

                # Otherwise if it is already in the open set
                if n_id in openset:
                    if openset[n_id].cost > node.cost:
                        openset[n_id].cost = node.cost
                        openset[n_id].pind = c_id

                else:
                    openset[n_id] = node

        rx, ry = self.calc_final_path(ngoal, graph_lookup)
        if graph_lookup == "reverse":
            self.r1_x = rx
            self.r1_y = ry
        elif graph_lookup == "forward":
            self.r2_x = rx
            self.r2_y = ry


if __name__ == "__main__":
    x, y = get_trajectory("../csv/data.csv")
    print("[INFO]: Read x and y from the csv file")
    coords = [(50.0, 20.0), (50.0, 40.0), (60.0, 40.0), (60.0, 20.0)]
    poly1 = Polygon(coords)
    obst = [poly1]
    rrt_dmp = RrtDmps(obstacles=obst, demo_x=x, demo_y=y)
    if rrt_dmp.cost_type == "dmp_traj":
        dmp = DMPs_discrete(n_dmps=2, n_bfs=100, dt=0.01)

        dmp.imitate_path(y_des=np.array([rrt_dmp.demo_x, rrt_dmp.demo_y]))
        dmp.y0[0] = rrt_dmp.start_x
        dmp.y0[1] = rrt_dmp.start_y
        dmp.goal[0] = rrt_dmp.goal_x
        dmp.goal[1] = rrt_dmp.goal_y

        rrt_dmp.dmp, rrt_dmp.dmp_dy, rrt_dmp.dmp_ddy, s = dmp.rollout()

    rrt_dmp.plot()
    if rrt_dmp.obstacles is not None:
        rrt_dmp.calc_obstacle_map()

    dmp_rev, dmp_rev_dy = rrt_dmp.reverse_dmp()

    time.sleep(10.0)
    p1 = Process(target=rrt_dmp.find_path, args=(rrt_dmp.start_x, rrt_dmp.start_y, rrt_dmp.dmp, rrt_dmp.dmp_dy))
    p1.start()

    p2 = Process(target=rrt_dmp.find_path, args=(rrt_dmp.goal_x, rrt_dmp.goal_y, dmp_rev, dmp_rev_dy, "forward"))
    p2.start()

    # rrt_dmp.find_path(rrt_dmp.start_x, rrt_dmp.start_y, rrt_dmp.dmp, rrt_dmp.dmp_dy)
    # rrt_dmp.find_path(rrt_dmp.goal_x, rrt_dmp.goal_y, dmp_rev, dmp_rev_dy, "forward")

    # time.sleep(1000000000.0)
    # plot, = plt.plot(rrt_dmp.r1_x, rrt_dmp.r1_y, "-r")
    # rrt_dmp.plots.append(plot)
    # rrt_dmp.legend.append('r1')
    #
    # plot, = plt.plot(rrt_dmp.r2_x, rrt_dmp.r2_y, "-r")
    # rrt_dmp.plots.append(plot)
    # rrt_dmp.legend.append('r2')

    plt.legend(rrt_dmp.plots, rrt_dmp.legend, loc='lower right')
    plt.show()
