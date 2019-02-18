#!/usr/bin/env python
import os
import sys
from utils import plot_path
from utils import get_trajectory
import time

bag_file = "/home/motion/lfd_ws/src/pydmps/bag/" + sys.argv[1]
print(bag_file)
os.system("rostopic echo -b " + bag_file + " -p /turtle1/pose > /home/motion/lfd_ws/src/pydmps/csv/data.csv")
time.sleep(4)
path_x, path_y = get_trajectory("../csv/data.csv")

plot_path(path_x, path_y, start=[3, 6], goal=[5, 10])

