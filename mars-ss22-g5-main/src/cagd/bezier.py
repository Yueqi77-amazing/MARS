#!/usr/bin/python
import math

import numpy
import numpy as np

from cagd.vec import vec2, vec3
from cagd.polyline import polyline
import copy


class bezier_curve:
    def __init__(self, degree):
        assert (degree >= 0)
        self.degree = degree
        self.control_points = [None for i in range(degree + 1)]
        self.color = "black"

    def set_control_point(self, index, val):
        assert (index >= 0 and index <= self.degree)
        self.control_points[index] = val

    def get_control_point(self, index):
        assert (index >= 0 and index <= self.degree)
        return self.control_points[index]

    # evaluates the curve at t
    def evaluate(self, t):
        return self.__de_casteljeau(t, 1)[0]

    # evaluates tangent at t
    def tangent(self, t):
        last_two_ctrl_pts = self.__de_casteljeau(t, 2)
        a = last_two_ctrl_pts[0]
        b = last_two_ctrl_pts[1]
        return b - a

    # calculates the normal at t
    def normal(self, t):
        pass

    # syntactic sugar so bezier curve can be evaluated as curve(t)
    # instead of curve.evaluate(t)
    def __call__(self, t):
        return self.evaluate(t)

    # calculates the de-casteljeau scheme until the column only has stop elements
    def __de_casteljeau(self, t, stop):
        assert (stop >= 1)
        column = self.control_points
        while len(column) > stop:
            new_column = [None for i in range(len(column) - 1)]
            for i in range(len(new_column)):
                new_column[i] = (1 - t) * column[i] + t * column[i + 1]
            column = new_column
        return column

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    # calculates the bezier representation of the derivative
    def get_derivative(self):
        pass

    def get_axis_aligned_bounding_box(self):
        min_vec = copy.copy(self.control_points[0])
        max_vec = copy.copy(self.control_points[0])
        for p in self.control_points:
            if p.x < min_vec.x:
                min_vec.x = p.x
            if p.y < min_vec.y:
                min_vec.y = p.y
            if p.x > max_vec.x:
                max_vec.x = p.x
            if p.y > max_vec.y:
                max_vec.y = p.y
        return (min_vec, max_vec)

    def draw(self, scene, num_samples):
        p0 = self(0)
        for i in range(1, num_samples + 1):
            t = i / num_samples
            p1 = self(t)
            scene.draw_line(p0, p1, self.color)
            p0 = p1

    def get_polyline_from_control_points(self):
        pl = polyline()
        for p in self.control_points:
            pl.append_point(p)
        return pl


class bezier_surface:
    # creates a bezier surface of degrees n,m
    # the degree parameter is a tuple (n,m)
    def __init__(self, degree):
        d1, d2 = degree
        assert (d1 >= 0 and d2 >= 0)
        self.degree = degree
        self.control_points = [[None for i in range(d2 + 1)] for j in range(d1 + 1)]
        white = (1, 1, 1)
        self.color = (white, white, white, white)
        self.curveature = (None, None, None, None)

    def set_control_point(self, index1, index2, val):
        assert (index1 >= 0 and index1 <= self.degree[0])
        assert (index2 >= 0 and index2 <= self.degree[1])
        self.control_points[index1][index2] = val

    def get_control_point(self, index1, index2):
        assert (index1 >= 0 and index1 <= self.degree[0])
        assert (index2 >= 0 and index2 <= self.degree[1])
        return self.control_points[index1][index2]

    def evaluate(self, t1, t2):
        return self.__de_casteljeau(t1, t2, (1, 1))[0][0]

    # sets the colors at the corners
    # c00 is the color at u=v=0, c01 is the color at u=0 v=1, etc
    # a color is a tuple (r,g,b) with values between 0 an 1
    def set_colors(self, c00, c01, c10, c11):
        self.color = (c00, c01, c10, c11)

    # sets the curveature at the corners
    # c00 is the curveature at u=v=0, c01 is the curveature at u=0 v=1, etc
    def set_curveature(self, c00, c01, c10, c11):
        self.curveature = (c00, c01, c10, c11)

    def __call__(self, t):
        t1, t2 = t
        return self.evaluate(t1, t2)

    def __de_casteljeau(self, t1, t2, stop):
        s1, s2 = stop
        d1, d2 = self.degree
        assert (s1 >= 1 and s2 >= 1)
        d1 += 1  # number of control points in each direction
        d2 += 1

        # apply the casteljeau scheme in one direction,
        # ie, reduce dimension from (d1, d2) to (s1, d2)
        column = self.control_points
        while d1 > s1:
            d1 -= 1
            new_column = [[None for i in range(d2)] for j in range(d1)]
            for i in range(d1):
                for j in range(d2):
                    new_column[i][j] = (1 - t1) * column[i][j] + t1 * column[i + 1][j]
            column = new_column

        # apply the casteljeau scheme in the other direction,
        # ie, reduce dimension from (s1, d2) to (s1, s2)
        while d2 > s2:
            d2 -= 1
            new_column = [[None for i in range(d2)] for j in range(d1)]
            for i in range(d1):
                for j in range(d2):
                    new_column[i][j] = (1 - t2) * column[i][j] + t2 * column[i][j + 1]
            column = new_column

        return column

    def normal(self, t1, t2):
        pass

    def get_derivative(self, direction):
        pass

    def subdivide(self, t1, t2):
        b0, b1 = self.__subdivide_u(t1)
        b00, b01 = b0.__subdivide_v(t2)
        b10, b11 = b1.__subdivide_v(t2)
        return [b00, b01, b10, b11]

    def __subdivide_u(self, t):
        du, dv = self.degree
        left = bezier_surface((du, dv))
        right = bezier_surface((du, dv))
        for k in range(du + 1):
            pts = self.__de_casteljeau(t, 0, (du - k + 1, dv + 1))
            left.control_points[k] = pts[0]
            right.control_points[-(k + 1)] = pts[-1]
        return (left, right)

    def __subdivide_v(self, t):
        du, dv = self.degree
        left = bezier_surface((du, dv))
        right = bezier_surface((du, dv))
        for k in range(dv + 1):
            pts = self.__de_casteljeau(0, t, (du + 1, dv - k + 1))
            for i in range(du + 1):
                left.control_points[i][k] = pts[i][0]
                right.control_points[i][-(k + 1)] = pts[i][-1]
        return (left, right)


class bezier_patches:
    CURVEATURE_GAUSSIAN = 0
    CURVEATURE_AVERAGE = 1
    CURVEATURE_PRINCIPAL_MAX = 2  # Maximale Hauptkruemmung
    CURVEATURE_PRINCIPAL_MIN = 3  # Minimale Hauptkruemmung
    COLOR_MAP_LINEAR = 4
    COLOR_MAP_CUT = 5
    COLOR_MAP_CLASSIFICATION = 6

    def __init__(self):
        self.patches: [bezier_surface] = []

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, p):
        return self.patches[p]

    def __setitem__(self, i, p):
        self.patches[i] = p

    def __delitem__(self, p):
        del self.patches[p]

    def __iter__(self):
        return iter(self.patches)

    def append(self, p):
        self.patches.append(p)

    # refines patches by subdividing each patch into four new patches
    # there are 4^num times more patches after calling this function
    def refine(self, num):
        for i in range(num):
            new_patches = bezier_patches()
            for p in self:
                new = p.subdivide(0.5, 0.5)
                for n in new:
                    new_patches.append(n)
            self.patches = new_patches

    def visualize_curveature(self, curveature_mode, color_map):
        # calculate curveatures at each corner point
        # set colors according to color map

        (m, n) = (3, 3)

        kappa_min = math.inf
        kappa_max = -math.inf
        for patch_i in range(len(self.patches)):
            patch = self.patches[patch_i]
            cps = patch.control_points

            b_u = [[self.delta10(cps, i, j) * m for j in range(n + 1)] for i in range(m)]
            b_v = [[self.delta01(cps, i, j) * n for j in range(n)] for i in range(m + 1)]
            b_uu = [[self.delta1010(cps, i, j) * m * (m - 1) for j in range(n + 1)] for i in range(m - 1)]
            b_uv = [[self.delta1001(cps, i, j) * m * n for j in range(n)] for i in range(m)]
            b_vv = [[self.delta0101(cps, i, j) * n * (n - 1) for j in range(n - 1)] for i in range(m + 1)]

            patch_kappas = [0 for _ in range(4)]

            for corner_i in range(4):
                b_u_c = self.get_corner(b_u, corner_i)
                b_v_c = self.get_corner(b_v, corner_i)
                b_uu_c = self.get_corner(b_uu, corner_i)
                b_uv_c = self.get_corner(b_uv, corner_i)
                b_vv_c = self.get_corner(b_vv, corner_i)

                E = b_u_c.dot(b_u_c)
                F = b_u_c.dot(b_v_c)
                G = b_v_c.dot(b_v_c)

                uXv = self.crossProduct(b_u_c, b_v_c)
                # uXvArray = numpy.cross((b_u.x,b_u.y,b_u.z), (b_v.x,b_v.y,b_v.z))
                # uXv = vec3(uXvArray[0], uXvArray[1], uXvArray[2])
                N = uXv * (1 / math.sqrt(uXv.dot(uXv)))

                e = N.dot(b_uu_c)
                f = N.dot(b_uv_c)
                g = N.dot(b_vv_c)

                kappa = None
                if curveature_mode == self.CURVEATURE_GAUSSIAN:
                    kappa = (e * g - f * f) / (E * G - F * F)
                elif curveature_mode == self.CURVEATURE_AVERAGE:
                    kappa = 0.5 * ((e * G - 2 * f * F + g * E) / (E * G - F * F))
                else:
                    raise "Error"

                patch_kappas[corner_i] = kappa

                if kappa < kappa_min:
                    kappa_min = kappa
                if kappa > kappa_max:
                    kappa_max = kappa

            patch.set_curveature(patch_kappas[0], patch_kappas[1], patch_kappas[2], patch_kappas[3])

        print(kappa_max)

        for patch in self.patches:
            patch_colors = [(0, 0, 0) for _ in range(4)]
            for corner_i in range(4):
                kappa = patch.curveature[corner_i]
                patch_colors[corner_i] = self.calc_color(color_map, kappa_max, kappa_min, kappa)
            patch.set_colors(patch_colors[0], patch_colors[1], patch_colors[2], patch_colors[3])

    def get_corner(self, square, num):
        match num:
            case 0:
                return square[0][0]
            case 1:
                return square[0][-1]
            case 2:
                return square[-1][0]
            case 3:
                return square[-1][-1]

    def calc_color(self, color_mapping, kappa_max, kappa_min, kappa):
        f: float
        match color_mapping:
            case self.COLOR_MAP_CUT:
                f = self.calcColorCut(kappa)
            case self.COLOR_MAP_LINEAR:
                f = self.calcColorLinear(kappa, kappa_min, kappa_max)
            case self.COLOR_MAP_CLASSIFICATION:
                f = self.calcColorClass(kappa)
            case _:
                raise "Error"
        # return (f,f,f)
        return self.h(f)

    def calcColorCut(self, kappa):
        return min(max(kappa, 0), 1)

    def calcColorLinear(self, kappa, kappa_min, kappa_max):
        return ((kappa - kappa_min) / (kappa_max - kappa_min)) #** (1/4)

    def calcColorClass(self, kappa):
        if (kappa < 0):
            return 0
        if (kappa == 0):
            return 0.5
        if (kappa > 0):
            return 1
        else:
            raise "Error"

    def h(self, x):
        if 0 <= x <= 0.25:
            return (0, 4 * x, 1)
        if 0.25 < x <= 0.5:
            return (0, 1, 2 - 4 * x)
        if 0.5 < x <= 0.75:
            return (4 * x - 2, 1, 0)
        if 0.75 < x <= 1:
            return (1, 4 - 4 * x, 0)
        else:
            raise "Error"

    def crossProduct(self, p1: vec3, p2: vec3):
        return vec3(p1.y * p2.z - p1.z * p2.y, p1.z * p2.x - p1.x * p2.z, p1.x * p2.y - p1.y * p2.x)

    def delta10(self, cps, i, j):
        return cps[i + 1][j] - cps[i][j]

    def delta01(self, cps, i, j):
        return cps[i][j + 1] - cps[i][j]

    def delta1010(self, cps, i, j):
        return cps[i + 2][j] - 2 * cps[i + 1][j] + cps[i][j]

    def delta1001(self, cps, i, j):
        return (cps[i + 1][j + 1] - cps[i + 1][j]) - (cps[i][j + 1] - cps[i][j])

    def delta0101(self, cps, i, j):
        return cps[i][j + 2] - 2 * cps[i][j + 1] + cps[i][j]

    def export_off(self):
        def export_point(p):
            return str(p.x) + " " + str(p.y) + " " + str(p.z)

        def export_colors(c):
            s = ""
            for x in c:
                s += str(x)
                s += " "
            s += "1"  # opacity
            return s

        s = "CBEZ333\n"
        for patch in self:
            # coordinates
            for row in patch.control_points:
                for p in row:
                    s += export_point(p)
                    s += "\n"

            # colors
            s += export_colors(patch.color[0])
            s += "\n"
            s += export_colors(patch.color[2])
            s += "\n"
            s += export_colors(patch.color[1])
            s += "\n"
            s += export_colors(patch.color[3])
            s += "\n"
            s += "\n"

        return s
