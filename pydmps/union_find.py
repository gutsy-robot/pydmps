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

    def find(self, p):
        """
        Find the set identifier for the item p.

        """
        # print("calling find on: ", p)
        id = self._id

        while p != id[p]:
            p = id[p] = id[id[p]]   # Path compression using halving.

        # print("find returned: ", p)
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
        print("sampled_pt id is: ", p)
        print("neighbor pt id is: ", q)
        id = self._id
        rank = self._rank

        i = self.find(p)
        j = self.find(q)

        print("sampled_pt parent is: ", i)
        print("neigbor pt parent is: ", j)
        if i == j:
            print("already in the same strongly connected component")
            return

        self._count -= 1
        print("one connected component would be reduced..")
        if rank[i] < rank[j]:
            print("neighbor id assigned to the sampled point")
            id[i] = j
        elif rank[i] > rank[j]:
            print("sampled pt id assigned to the neighbor")
            id[j] = i
        else:
            # print("neighbor id assigned to the sampled point")
            id[i] = j
            rank[j] += 1

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

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._id])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"


# uf = UF(10)
#
# print("after initialising count is: ", uf.count())
# print("after initialising id is: ", uf._id)
# print("after initialising rank is: ", uf._rank)
#
#
# uf.union(0, 1)
# uf.union(3, 1)
# uf.union(4, 1)
# uf.union(5, 1)
#
# print("after 4 union operations count is: ", uf.count())
# print("after 4 union operations id is: ", uf._id)
# print("after 4 union operations rank is: ", uf._rank)
#
# uf.union(6, 2)
# uf.union(7, 2)
# uf.union(8, 2)
# uf.union(9, 2)
#
# print("after first set of unions scc is: ", uf.get_scc())
#
# uf.union(0, 7)
# uf.union(4, 7)
# uf.union(5, 8)
# print("after final union operations count is: ", uf.count())
# print("after final union operations id is: ", uf._id)
# print("after final union operations rank is: ", uf._rank)
#
# print("--------")
# print("scc after all operations is: ", uf.get_scc())


#
# uf.add(1000)
# uf.union(3, 1000)
# print("union 1000 and 3")
# print("after adding 3")
# print("rank is: ", uf._rank)
# print("id is: ", uf._id)
# print("count is: ", uf.count())
