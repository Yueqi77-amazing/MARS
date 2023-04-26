#!/usr/bin/python

from cagd.polyline import polyline
from cagd.spline import spline, spline_surface, knots
from cagd.bezier import bezier_surface, bezier_patches
from cagd.vec import vec2, vec3
import cagd.scene_2d as scene_2d


pts = [ vec2(0.05,5.5),
        vec2(1.5,5),
        vec2(2,4),
        vec2(1.7,2.5),
        vec2(0.7,1.8),
        vec2(2,1.3),
        vec2(2,0.9),
        vec2(1.2,0.8),
        vec2(.7,0.4),
        vec2(.7,-1),
        vec2(.7,-2.8),
        vec2(2,-4),
        vec2(2,-4.6),]
spl = spline(3)
spl = spl.interpolate_cubic(spline.INTERPOLATION_CHORDAL, pts)
#you can activate these lines to view the input spline
#spl.set_color("#0000ff")
#sc = scene_2d.scene()
#sc.set_resolution(900)
#sc.add_element(spl)
#sc.write_image()
#sc.show()

surface = spl.generate_rotation_surface(50)

# kns = ([0,0,0,1,2,3,4,5,5,5],[0,0,0,1,2,3,4,5,6,7,8,9,10,11,11,11])
# surface.knots = kns

for cp in surface.control_points:
        print([(round(p.x, 1), round(p.y, 1), round(p.z, 1)) for p in cp])

bezier_patches = surface.to_bezier_patches()

path = "surfaces.off"
f = open(path, 'w')
f.write(bezier_patches.export_off())
f.close()

import os
os.chdir('rover-0.1.1')
os.system('cargo run --release -- ../surfaces.off')
