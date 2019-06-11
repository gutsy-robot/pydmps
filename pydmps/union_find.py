#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements an union find or disjoint set data structure.

An union find data structure can keep track of a set of elements into a number
of disjoint (nonoverlapping) subsets. That is why it is also known as the
disjoint set data structure. Mainly two useful operations on such a data
structure can be performed. A *find* operation determines which subset a
particular element is in. This can be used for determining if two
elements are in the same subset. An *union* Join two subsets into a
single subset.

The complexity of these two operations depend on the particular implementation.
It is possible to achieve constant time (O(1)) for any one of those operations
while the operation is penalized. A balance between the complexities of these
two operations is desirable and achievable following two enhancements:

1.  Using union by rank -- always attach the smaller tree to the root of the
    larger tree.
2.  Using path compression -- flattening the structure of the tree whenever
    find is used on it.

complexity:
    * find -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
      `inverse ackerman function
      <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.
    * union -- :math:`O(\\alpha(N))` where :math:`\\alpha(n)` is
      `inverse ackerman function
      <http://en.wikipedia.org/wiki/Ackermann_function#Inverse>`_.

"""
from copy import deepcopy

class UF:
    """

    An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.

    """

    def __init__(self, N):
        """
        Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.

        """
        # id for each node
        self._id = list(range(N))

        # number of connected components
        self._count = N

        # node with a greater rank is bigger
        self._rank = [0] * N

    def find(self, p, show=False):
        """
        Find the set identifier for the item p.

        """
        # print("calling find on: ", p)
        id = self._id
        # if show:
        #     print("starting with: ", p)

        while p != id[p]:
            # p = id[p] = id[id[p]]   # Path compression using halving.
            p = id[p]
            # if show:
            #     print("p inside the loop is: ", p)

        # print("find returned: ", p)
        # if show:
        #     print("returning: ", p)
        #     print("id[p]: ", id[p])
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""
        # print("calling union")
        # print("sampled_pt id is: ", p)
        # print("neighbor pt id is: ", q)
        # initial_nscc = len(self.get_scc().keys())
        # print("initial number of nscc: ", initial_nscc)
        # print("initial count is: ", self.count())
        # print("initial id array(before find p) is: ", self._id)
        id = self._id
        rank = self._rank
        # print("calling find on: ", p)
        i = self.find(p, show=True)

        # print("calling find on: ", q)
        # print("id array before find q is: ", self._id)
        j = self.find(q, show=True)

        # print("sampled_pt parent is: ", i)
        # print("neigbor pt parent is: ", j)
        if i == j:
            # print("already in the same strongly connected component")
            return

        self._count -= 1
        # print("one connected component would be reduced..")
        if rank[i] < rank[j]:
            # print("neighbor id assigned to the sampled point")
            id[i] = j
        elif rank[i] > rank[j]:
            # print("sampled pt id assigned to the neighbor")
            id[j] = i
        else:
            # print("neighbor id assigned to the sampled point")
            id[i] = j
            rank[j] += 1

        # if id != self._id:
        #     print("WE HAVE A BIGGER PROBLEM HERE..")

        # final_nscc = len(self.get_scc().keys())
        # if final_nscc - initial_nscc != -1:
        #     print("ERROR IN UNIONING: ", (p, q))
        #     print("difference in scc is: ", final_nscc - initial_nscc)
        #     print("find for p returned: ", i)
        #     print("find for q returned: ", j)
        #     # print("id array before:  ", id_array_before)
        #     print("id array later: ", self._id)

    def add(self, p):
        self._id.append(p)
        self._count += 1
        self._rank.append(0)

    def get_num_nodes(self):
        return len(self._id)

    def get_scc(self):
        scc = {}

        for x in self._id:
            x_parent = self.find(x)
            if x_parent in scc.keys():
                scc[x_parent].append(x)

            else:
                scc[x_parent] = [x]

        return scc

    def get_info(self):
        return self._id, self._rank, self._count

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"


# uf = UF(2)
#
# uf.add(2)
# uf.add(3)
# uf.add(4)
#
# print("id and rank are: ", uf.get_info())
#
# uf.union(0, 1)
# print("after union 0 and 1...")
# print("id and rank are: ", uf.get_info())
#
#
# uf.union(3, 2)
# print("after union 2 and 3...")
# print("id and rank are: ", uf.get_info())
#
# uf.union(1, 4)
# print("after union 1 and 4...")
# print("id and rank are: ", uf.get_info())
#
# uf.union(1, 2)
# print("after union 1 and 2...")
# print("id and rank are: ", uf.get_info())
#
# uf.union(2, 1)
# print("after union 2 and 1...")
# print("id and rank are: ", uf.get_info())


