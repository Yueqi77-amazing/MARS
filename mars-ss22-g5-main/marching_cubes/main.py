import math
from enum import Enum

import numpy

from cube import CubeVertices, CubeEdges, CubeEdgeFlags, CubeTriangles, CubeVertexDirections


class Shape(Enum):
    SPHERE = 1
    OCTAHEDRON = 2
    CUBE = 3
    TORUS = 4


class SubCube:
    corner_polarities = ""
    edge_intersects = ""

    triangles = []

    def __init__(self, corns):
        self.corners = corns

    def calc_intersects(self, f, I):
        corner_bits = ""
        for corner in self.corners:
            corner_bits += ("1" if (f(corner) < I) else "0")
        edge_bits = bin(CubeEdgeFlags[int(corner_bits, 2)])

        corner_bits = corner_bits[::-1]
        edge_bits = edge_bits[::-1]

        self.corner_polarities = corner_bits
        self.edge_intersects = edge_bits

    def calc_triangles(self, F, I):
        temp = CubeTriangles[int(self.corner_polarities, 2)]

        i = 0
        while i < 16 and temp[i] != -1:
            c1 = self.get_edge_coords(temp[i + 0], F, I)
            c2 = self.get_edge_coords(temp[i + 1], F, I)
            c3 = self.get_edge_coords(temp[i + 2], F, I)
            self.triangles.append([c1, c2, c3])
            i += 3

    def get_edge_coords(self, edge_i, F, I):
        corner1 = CubeEdges[edge_i][0]
        corner2 = CubeEdges[edge_i][1]

        p1 = self.corners[corner1]
        p2 = self.corners[corner2]

        v1 = F(p1) - I
        v2 = F(p2) - I

        q = [0, 0, 0]
        for i in range(3):
            q[i] = (v1 * p2[i] - v2 * p1[i]) / (v1 - v2)

        return q


def sphere_func(corner):
    (x, y, z) = corner
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def octahedron_func(corner):
    (x, y, z) = corner
    return abs(x) + abs(y) + abs(z)


def cube_func(corner):
    (x, y, z) = corner
    return max(abs(x), abs(y), abs(z))


def torus_func(corner):
    (x, y, z) = corner
    return (x ** 2 + y ** 2 + z ** 2 + 0.375) ** 2 - 2 * (x ** 2 + y ** 2)


def marching_cubes(n, L, I, s):
    # translate the cube so that it is centered at the origin and that it has length L
    cube = []
    for corner in CubeVertices:
        cube.append([(coord - 0.5) * 2 * L for coord in corner])
    cubes = subdivide_cube(n, L)

    match s:
        case Shape.SPHERE:
            func = sphere_func
        case Shape.OCTAHEDRON:
            func = octahedron_func
        case Shape.CUBE:
            func = cube_func
        case Shape.TORUS:
            func = torus_func
        case _:
            raise "eroor"

    triangles = []

    corners = []

    for cube in cubes:
        cube.calc_intersects(func, I)
        cube.calc_triangles(func, I)
        triangles.extend(cube.triangles)
        for corner_i in range(len(cube.corners)):
            if cube.corner_polarities[-1 - corner_i] == "1":
                corners.append(cube.corners[corner_i])

    print(triangles)

    polarities = []


def subdivide_cube(n, L):
    length = 2 * L / n
    scaled_unit_square = [[coord * length for coord in corner] for corner in CubeVertices]
    cubes = []

    for x in numpy.arange(-L, L, length):
        for y in numpy.arange(-L, L, length):
            for z in numpy.arange(-L, L, length):
                cube = SubCube([[x + corner[0], y + corner[1], z + corner[2]] for corner in scaled_unit_square])
                cubes.append(cube)
    return cubes


if __name__ == "__main__":
    # n = input("Enter n ")
    # L = input("Enter L ")
    # I = input("Enter I ")
    # s_num = input("Enter shape 1: Sphere, 2: Octahedron, 3: Cube, 4: Torus ")
    # s = s_num

    marching_cubes(4, 1, 1, Shape.SPHERE)
