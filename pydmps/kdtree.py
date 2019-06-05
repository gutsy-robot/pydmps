import scipy.spatial
import numpy as np


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

#
# x = [1, 2, 3, 4, 5, 6]
# y = [1, 2, 3, 4, 5, 6]
# t = [1, 2, 3, 4, 5, 6]
#
# tree = KDTree(np.vstack((x, y, t)).T)
# ind, d = tree.search(np.array([0, 0, 0]), 1)
#
# print("index of the node closest to (0, 0) is: ", ind)
# print("distance of the point closest to (0, 0) is: ", d)
#
# print(tree.tree.data[-1][2])

