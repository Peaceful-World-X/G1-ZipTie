"""A stationary pile of curved spare ties in the far half of the left tray."""
import numpy as np
from build_scene import ROOT, element


def add_tray_stock(root, tray):
    rng = np.random.default_rng(405)
    vertices, triangles = [], []
    # Keep the near end clear for the dynamically deposited workpiece.
    for layer in range(16):
        center = rng.uniform([.035, -.045], [.065, .045])
        start = rng.uniform(0, 2*np.pi)
        angle = np.linspace(start, start+rng.uniform(3.2, 5.6), 40)
        line = center + np.column_stack((.077*np.cos(angle), .034*np.sin(angle)))
        normal = np.column_stack((-.034*np.cos(angle), -.077*np.sin(angle)))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        skew = rng.uniform(-.56, .56)
        planar = np.array([[np.cos(skew), -np.sin(skew)], [np.sin(skew), np.cos(skew)]])
        line = center + (line-center) @ planar.T
        normal = normal @ planar.T
        base = len(vertices)
        bottom = .006 + .00048*layer + .0005*(1+np.sin(2*angle))
        for i, (p, n, z) in enumerate(zip(line, normal, bottom)):
            # A thicker first section represents the head of each spare tie.
            width = .003 if i < 2 else .00145
            height = .003 if i < 2 else .0008
            for sign, dz in ((-1,0), (1,0), (1,height), (-1,height)):
                vertices.append((*list(p + sign*width*n), z+dz))
            if i:
                last = base + (i-1)*4
                for j in range(4):
                    k = (j+1) % 4
                    triangles.extend(((last+j,last+k,last+4+k),(last+j,last+4+k,last+4+j)))
        end = base + 4*(len(line)-1)
        triangles.extend(((base+2,base+1,base),(base+3,base+2,base),
                          (end,end+1,end+2),(end,end+2,end+3)))
    path = ROOT/'assets/workcell/tray_stock.obj'
    with path.open('w', encoding='ascii') as out:
        for v in vertices: out.write('v %.7f %.7f %.7f\n' % v)
        for a,b,c in triangles: out.write(f'f {a+1} {b+1} {c+1}\n')
    element(root.find('asset'), 'mesh', name='spare_ties', file=str(path))
    element(tray, 'geom', type='mesh', mesh='spare_ties', rgba='.94 .62 .095 1',
            mass='0', contype='0', conaffinity='0', group='2')
