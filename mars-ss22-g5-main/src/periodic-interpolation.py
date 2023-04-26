#!/usr/bin/python
import math

import cagd.scene_2d as scene
from cagd.vec import vec2
from cagd.spline import spline, knots, norm
from cagd.polyline import polyline


# returns a list of num_samples points that are uniformly distributed on the unit circle
def unit_circle_points(num_samples):
    angle = 2 * math.pi / num_samples

    points = [vec2(math.cos(i * angle), math.sin(i * angle)) for i in range(num_samples)]

    return points


# calculates the deviation between the given spline and a unit circle
def calculate_circle_deviation(spline: spline):
    num = len(spline.knots)

    (a, b) = spline.support()
    spline_point: vec2
    unit_point: vec2

    spline_point = spline.evaluate(a + ((b - a) / (num * 2)))
    unit_point = vec2(math.cos(math.pi / num), math.sin(math.pi / num))

    return norm(spline_point.__sub__(unit_point))


# interpolate 6 points with a periodic spline to create the number "8"
pts = [vec2(0, 2.5), vec2(-1, 1), vec2(1, -1), vec2(0, -2.5), vec2(-1, -1), vec2(1, 1)]
example_spline = spline(3)
s = example_spline.interpolate_cubic_periodic(pts)
s.set_color("green")
p = s.get_polyline_from_control_points()
p.set_color("blue")
sc = scene.scene()
sc.set_resolution(900)
sc.add_element(s)
sc.add_element(p)

# generate a spline that approximates the unit circle
n = 8
circle_pts = unit_circle_points(n)
circle = example_spline.interpolate_cubic_periodic(circle_pts)
sc.add_element(circle)
calculate_circle_deviation(circle)

sc.write_image()
sc.show()


# if __name__ == "__main__":
#     print([(p.x, p.y) for p in unit_circle_points(4)])
