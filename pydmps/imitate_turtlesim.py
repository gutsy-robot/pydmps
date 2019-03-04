#!/usr/bin/env python
import numpy as np
import os
import sys
from utils import plot_path
from utils import get_trajectory
import time
from shapely.geometry.polygon import LinearRing, Polygon

bag_file = "/home/motion/lfd_ws/src/pydmps/bag/" + sys.argv[1]
print(bag_file)
os.system("rostopic echo -b " + bag_file + " -p /turtle1/pose > /home/motion/lfd_ws/src/pydmps/csv/data.csv")
time.sleep(4)
path_x, path_y = get_trajectory("../csv/data.csv")

# path_x = [5.0] * 20 + list(np.linspace(5.0, 8.0, 20))
# path_y = list(np.linspace(0.0, 3.0, 20)) + [3.0] * 20

coords = [(0.0, 2.0), (0.0, 0.0), (2.0, 0.0), (2.2, 2.2)]
poly = Polygon(coords)
plot_path(path_x, path_y, start=[5.0, 5.0], goal=[7.0, 7.0], obstacles=[poly])



