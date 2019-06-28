from prm_ucb2 import PRM_planning
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon, LineString
from utils import get_trajectory, check_collision, avoid_obstacles, sample_dmp_normal, sample_uniform, ind_max
from dmp_discrete import DMPs_discrete
import os
from kdtree import KDTree
from statistics import mean, stdev
import json
import scipy.interpolate
import math
from mpl_toolkits.mplot3d import axes3d


main_dir = "test_new_additions_lazy_one_obstacle/"
try:
    os.mkdir(main_dir)

except:
    print("destination folder exists")

x, y = get_trajectory("../csv/data.csv")
scaled_x = [0.01 * 10 * i for i in x]
scaled_y = [0.01 * 10 * j for j in y]

x2 = [0.0, 0.1, 0.5]
y2 = [0.0, 0.5, 0.8]

poly_curve = scipy.interpolate.BarycentricInterpolator(x2, y2)
poly_x, poly_y = [], []

for i in range(0, 80):
    poly_x.append(i * 0.01)
    poly_y.append(poly_curve(i * 0.01))


path_x = scaled_x
path_y = scaled_y

edge_resolution_factor = 2.0
neighbor_radius_factor = 10.0
use_obstacle_cost = False
use_ucb = True
num_goal_pts = 50
uniform_only = False
normal_only = False
dynamic_radius = False
num_points = 1000
plot_sampled = True
plot_roadmap = False
obstacle_pot = 0.1
incremental_reward_weight = 1.0
connectivity_reward_weight = 1.0
lazy_collision_check = True
new_scc_reward = 1.0
scc_connect_reward = 1.5

use_discretised_cost = True
dmp_normal_cov = 0.04
uniform_max = 1.2
uniform_min = 0.8
# earlier this was 2.0
uniform_max_t = 2.0

inputs = {
    'edge_resolution_factor': edge_resolution_factor,
    'neighbor_radius_factor': neighbor_radius_factor,
    'use_obstacle_cost': use_obstacle_cost,
    'use_ucb': use_ucb,
    'num_goal_pts': num_goal_pts,
    'uniform_only': uniform_only,
    'normal_only': normal_only,
    'dynamic_radius': dynamic_radius,
    'num_points': num_points,
    'plot_sampled':  plot_sampled,
    'plot_roadmap': plot_roadmap,
    'obstacle_pot': obstacle_pot,
    'incremental_reward_weight': incremental_reward_weight,
    'connectivity_reward_weight' : connectivity_reward_weight,
    'use_discretised_cost': use_discretised_cost,
    'dmp_normal_cov': dmp_normal_cov,
    'uniform_max': uniform_max,
    'uniform_min': uniform_min,
    'uniform_max_t': uniform_max_t,
    'scc_connect_reward' : scc_connect_reward,
    'new_scc_reward' : new_scc_reward,
    'lazy_collision_check' : lazy_collision_check
}

with open(main_dir + 'inputs.json', 'w') as fp:
    json.dump(inputs, fp, indent=4)

sx = 10.0 * 0.01
sy = 10.0 * 0.01

gx = 75.0 * 0.01
gy = 70.0 * 0.01

# coords = [(50.0 * 0.01, 30.0 * 0.01), (50.0 * 0.01, 40.0 * 0.01), (60.0 * 0.01, 40.0 * 0.01),
#           (60.0 * 0.01, 30.0 * 0.01)]
# poly1 = Polygon(coords)
#
# coords2 = [(25.0 * 0.01, 15.0 * 0.01), (25.0 * 0.01, 25.0 * 0.01), (35.0 * 0.01, 25.0 * 0.01),
#            (35.0 * 0.01, 15.0 * 0.01)]
# poly2 = Polygon(coords2)
#
# coords3 = [(70.0 * 0.01, 30.0 * 0.01), (70.0 * 0.01, 40.0 * 0.01), (80.0 * 0.01, 40.0 * 0.01),
#           (80.0 * 0.01, 30.0 * 0.01)]
# poly3 = Polygon(coords3)
#
# coords4 = [(50.0 * 0.01, 50.0 * 0.01), (50.0 * 0.01, 60.0 * 0.01), (60.0 * 0.01, 60.0 * 0.01),
#           (60.0 * 0.01, 50.0 * 0.01)]
# poly4 = Polygon(coords4)

coords = [(50.0 * 0.01, 30.0 * 0.01), (50.0 * 0.01, 45.0 * 0.01), (60.0 * 0.01, 45.0 * 0.01),
          (60.0 * 0.01, 30.0 * 0.01)]
poly1 = Polygon(coords)

coords2 = [(25.0 * 0.01, 25.0 * 0.01), (25.0 * 0.01, 40.0 * 0.01), (35.0 * 0.01, 40.0 * 0.01),
           (35.0 * 0.01, 25.0 * 0.01)]
poly2 = Polygon(coords2)

coords3 = [(70.0 * 0.01, 30.0 * 0.01), (70.0 * 0.01, 40.0 * 0.01), (150.0 * 0.01, 40.0 * 0.01),
          (150.0 * 0.01, 30.0 * 0.01)]
poly3 = Polygon(coords3)

coords4 = [(90.0 * 0.01, 50.0 * 0.01), (90.0 * 0.01, 70.0 * 0.01), (100.0 * 0.01, 70.0 * 0.01),
          (100.0 * 0.01, 50.0 * 0.01)]
poly4 = Polygon(coords4)

coords5 = [(90.0 * 0.01, 20.0 * 0.01), (90.0 * 0.01, 40.0 * 0.01), (150.0 * 0.01, 40.0 * 0.01),
          (150.0 * 0.01, 20.0 * 0.01)]
poly5 = Polygon(coords4)

obstacles = [poly1]
path_costs = []

edge_collision_check_time_total = []
reward_calc_time_total = []
sampling_time_total = []
dijkstra_time_total = []

num_runs = 3

for i in range(0, num_runs):
    print("starting run number: ", i)
    data_path = main_dir + str(i) + "/"

    try:
        os.mkdir(data_path)

    except:
        print("destination folder exists")

    show_animation = True
    fig, plt2d = plt.subplots()

    fig_dij, plt_dij = plt.subplots()

    fig2, plt_mean_reward = plt.subplots()
    # plt_mean_reward = fig2.add_subplot(111)

    fig2.suptitle('Mean Reward', fontsize=24)

    fig4 = plt.figure()
    ax = fig4.add_subplot(111, projection='3d')

    fig_3Dcost = plt.figure()
    ax_3Dcost = fig_3Dcost.add_subplot(111, projection='3d')

    fig_3Dnodes = plt.figure()
    ax_3Dnodes = fig_3Dnodes.add_subplot(111, projection='3d')

    fig5, plt_ucb = plt.subplots()
    # plt_ucb = fig5.add_subplot(111)
    fig5.suptitle('UCB Values', fontsize=24)

    fig6, plt_connected = plt.subplots()
    # plt_connected = fig6.add_subplot(111)
    fig6.suptitle('Number of disconnected components', fontsize=24)

    fig_fraction_plot, plt_fraction = plt.subplots()
    fig_fraction_plot.suptitle('Arm Fraction', fontsize=24)

    if plot_roadmap:
        fig7, plt_roadmap = plt.subplots()
        # plt_roadmap = fig7.add_subplot(111)
        fig7.suptitle('Roadmap plot', fontsize=24)

    else:
        plt_roadmap = None

    # scaled_x = [0.01 * 10 * i for i in x]
    # scaled_y = [0.01 * 10 * j for j in y]

    z = np.zeros((1, len(path_x)))

    orig_path = ax.plot_wireframe(np.array(path_x), np.array(path_y), z, '--', linewidth=2)
    ax.scatter(path_x[0], path_y[0], z[0][0])
    ax.scatter(path_x[-1], path_y[-1], z[0][-1])
    ax_3Dnodes.scatter(path_x[0], path_y[0], z[0][0])
    ax_3Dnodes.scatter(path_x[-1], path_y[-1], z[0][-1])
    plt2d.plot(path_x, path_y, '--', linewidth=2, label='demonstration')
    plt2d.scatter(path_x[0], path_y[0])
    plt2d.annotate("st. original", (path_x[0], path_y[0]))
    plt2d.scatter(path_x[-1], path_y[-1])
    plt2d.annotate("f original.", (path_x[-1], path_y[-1]))
    ax.scatter(sx, sy, 0.0)
    ax_3Dcost.scatter(sx, sy, 0.0)
    plt2d.scatter(sx, sy)
    plt2d.annotate("new_st", (sx, sy))
    plt2d.scatter(gx, gy)
    plt2d.annotate("new_fin", (gx, gy))

    plt_dij.plot(path_x, path_y, '--', linewidth=2, label='demonstration')
    plt_dij.scatter(path_x[0], path_y[0])
    plt_dij.annotate("st. original", (path_x[0], path_y[0]))
    plt_dij.scatter(path_x[-1], path_y[-1])
    plt_dij.annotate("f original.", (path_x[-1], path_y[-1]))
    # ax.scatter(sx, sy, 0.0)
    plt_dij.scatter(sx, sy)
    plt_dij.annotate("new_st", (sx, sy))
    plt_dij.scatter(gx, gy)
    plt_dij.annotate("new_fin", (gx, gy))

    plt2d.plot(sx, sy, "xr")
    plt2d.plot(gx, gy, "xb")
    plt2d.grid(True)
    plt2d.axis("equal")

    plt_dij.plot(sx, sy, "xr")
    plt_dij.plot(gx, gy, "xb")
    plt_dij.grid(True)
    plt_dij.axis("equal")

    dmp = DMPs_discrete(n_dmps=2, n_bfs=100, dt=0.01, run_time=1.0)
    dmp.imitate_path(y_des=np.array([path_x, path_y]))
    # dmp.imitate
    dmp.y0[0] = sx
    dmp.y0[1] = sy
    dmp.goal[0] = gx
    dmp.goal[1] = gy

    y_track_nc, dy_track_nc, ddy_track_nc, s = dmp.rollout()
    print("[INFO]: trajectory rolled out successfully..")
    print("shape of y_track_nc is: ", y_track_nc.shape)
    y_track_nc_time = []
    print("total time is: ", len(y_track_nc) * dmp.dt)
    for w in range(0, len(y_track_nc)):
        y_track_nc_time.append((w + 1) * dmp.dt)

    print("end time in the time array is: ", y_track_nc_time[-1])
    y_track_nc_time = np.array([y_track_nc_time])
    y_track_nc_x = np.array(y_track_nc[:, 0])
    y_track_nc_y = np.array(y_track_nc[:, 1])

    ax.scatter(y_track_nc_x, y_track_nc_y, y_track_nc_time, s=1.0)
    ax_3Dnodes.scatter(y_track_nc_x, y_track_nc_y, y_track_nc_time)

    plot_2d_dmp, = plt2d.plot(y_track_nc_x, y_track_nc_y, label='dmp')
    plt2d.scatter(y_track_nc_x, y_track_nc_y, s=8.0)

    plt_dij.plot(y_track_nc_x, y_track_nc_y, label='dmp')
    plt_dij.scatter(y_track_nc_x, y_track_nc_y)

    dmp_time_para = []
    dmp_dy_time_para = []
    for k in range(0, len(y_track_nc)):
        temp = list(y_track_nc[k])
        temp_vel = list(dy_track_nc[k])
        temp.append((k + 1) * dmp.dt)
        temp_vel.append((k + 1) * dmp.dt)
        dmp_dy_time_para.append(temp_vel)
        dmp_time_para.append(temp)

    dmp_time_para = np.array(dmp_time_para)
    dmp_dy_time_para = np.array(dmp_dy_time_para)

    dmp_time_kdtree = KDTree(np.vstack((dmp_time_para[:, 0], dmp_time_para[:, 1], dmp_time_para[:, 2])).T)

    rx, ry, rt, path_cost, cost_array, edge_check_time_sampling, reward_calc_time, total_sampling_time, dijkstra_time = \
        PRM_planning(
                     sx, sy, gx, gy, obstacles=obstacles, guiding_paths=[dmp_time_kdtree],
                     dmp_vel=dmp_dy_time_para, guiding_path_weights=[1.0],
                     incremental_reward_weight=incremental_reward_weight,
                     connectivity_reward_weight=connectivity_reward_weight,
                     use_ucb=use_ucb, uniform_only=uniform_only, normal_only=normal_only,
                     edge_resolution_factor=edge_resolution_factor,
                     use_obstacle_cost=use_obstacle_cost,
                     neighbor_radius_factor=neighbor_radius_factor, num_goal_pts=num_goal_pts,
                     dynamic_radius=dynamic_radius,
                     num_points=num_points, obstacle_pot=obstacle_pot,
                     plot_sampled=plot_sampled, plot_roadmap=plot_roadmap,
                     use_discretised_cost=use_discretised_cost, dmp_normal_cov=dmp_normal_cov,
                     plt2d=plt2d, plt_mean_reward=plt_mean_reward, plt_ucb=plt_ucb,
                     plt_connected=plt_connected, plt_roadmap=plt_roadmap, uniform_max=uniform_max,
                     uniform_min=uniform_min, uniform_max_t=uniform_max_t, plt_dij=plt_dij,
                     plt_nodes=ax_3Dnodes, plt_fraction=plt_fraction, lazy_collision_check=lazy_collision_check)

    print("end time is: ", rt[0])
    print("path cost is: ", path_cost)
    path_costs.append(path_cost)
    num_nodes_path = len(rx)

    goal_time = rt[0]

    rx = np.array(rx)
    ry = np.array(ry)
    rt = np.array([rt])

    path_length = 0.0

    total_edge_collision_time = sum(edge_check_time_sampling)
    total_reward_calc_time = sum(reward_calc_time)

    reward_calc_time_total.append(total_reward_calc_time)
    edge_collision_check_time_total.append(total_edge_collision_time)
    sampling_time_total.append(total_sampling_time)
    dijkstra_time_total.append(dijkstra_time)

    for j in range(0, len(rx)):
        if j < len(rx) - 1:
            path_length += math.sqrt((ry[j+1] - ry[j]) ** 2 + (rx[j+1] - rx[j]) ** 2)

        plt_dij.scatter(rx[j], ry[j])
        plt_dij.annotate(str(round(cost_array[j], 3)), (rx[j], ry[j]))

    cost_array = np.array([cost_array])
    plot = ax.plot_wireframe(rx, ry, rt, color='k', label='ucb')
    ax.legend()
    ax_3Dcost.plot_wireframe(rx, ry, cost_array, color='k', label='ucb')

    plt2d.plot(rx, ry, color='k', label='ucb path1')
    plt2d.legend()
    plt_dij.plot(rx, ry, color='k', label='ucb path1')
    plt_dij.legend()

    title = "path cost: " + str(path_cost) + "__path_nodes: " + str(num_nodes_path) + "__g_time: " + "\n" + \
            str(goal_time) + "__run: " + str((i + 1)) + "__path_length: " + str(path_length)

    if obstacles is not None:
        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            plt2d.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
            plt_dij.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    fig.suptitle(title)
    fig.savefig(data_path + "plt2d.png")
    fig_dij.savefig(data_path + "plt_dij.png")
    fig2.savefig(data_path + "plt_mean_reward.png")
    fig4.savefig(data_path + "plt3d.png")
    fig5.savefig(data_path + "plt_ucb.png")
    fig6.savefig(data_path + "plt_connected.png")
    fig_3Dcost.suptitle('Cost Curve', fontsize=24)
    fig_3Dcost.savefig(data_path + "plt_cost.png")
    fig_3Dnodes.suptitle('nodes', fontsize=24)
    fig_3Dnodes.savefig(data_path + "plt_3Dnodes.png")
    fig_fraction_plot.savefig(data_path + "plt_fraction.png")

print("path cost is: ", mean(path_costs))
print("path cost array is: ", path_costs)

output = {
    'num_runs': num_runs, 'mean_cost': mean(path_costs),
    'path_cost_dev': stdev(path_costs),
    'mean_sampling_time': mean(sampling_time_total),
    'sampling_time_dev': stdev(sampling_time_total),
    'mean_dij_time': mean(dijkstra_time_total),
    'dij_time_dev' : stdev(dijkstra_time_total),
    'mean_edge_collision_time': mean(edge_collision_check_time_total),
    'edge_collision_time_dev': stdev(edge_collision_check_time_total),
    'mean_state_reward_time': mean(reward_calc_time_total),
    'state_reward_time_dev': stdev(reward_calc_time_total)}

with open(main_dir + 'outputs.json', 'w') as fp:
    json.dump(output, fp, indent=4)

