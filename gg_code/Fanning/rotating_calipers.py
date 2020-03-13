from __future__ import generators
from scipy import spatial


# # # Copyright (c) 2017 ActiveState Software Inc.

# # # Permission is hereby granted, free of charge, to any person obtaining a 
# # # copy of this software and associated documentation files (the "Software"), 
# # # to deal in the Software without restriction, including without limitation 
# # # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# # # and/or sell copies of the Software, and to permit persons to whom the 
# # # Software is furnished to do so, subject to the following conditions:

# # # The above copyright notice and this permission notice shall be included 
# # # in all copies or substantial portions of the Software.

# # # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# # # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# # # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# # # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# # # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# # # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# # # IN THE SOFTWARE.

# # # Originally published: 2002-03-07 17:56:08
# # # Last updated: 2003-10-16 18:44:57
# # # Author: David Eppstein
# # # https://github.com/ActiveState/code/tree/master/recipes/Python/117225_Convex_hull_diameter_2d_point


def min_width(points):
    '''Given a list of 2d points, returns the pair that represent the minimum width.'''
    diam = min([spatial.distance.euclidean(p,q)
                     for p,q in rotatingCalipers(points)])
    return diam


def max_width(points):
    '''Given a list of 2d points, returns the pair that represent the minimum width.'''
    diam = max([spatial.distance.euclidean(p,q)
                    for p,q in rotatingCalipers(points)])
    return diam


def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])


def convex_hull(points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    points.sort()
    for p in points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L


def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = convex_hull(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1