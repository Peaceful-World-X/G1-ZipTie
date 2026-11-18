"""Visual geometry and lighting for the 24-second G1 demonstration."""
import numpy as np
from build_scene import ROOT, element, vec
from workcell_textures import make_textures

VISUAL = dict(contype='0', conaffinity='0', mass='0', group='2')


def finish(geom, material):
    geom.attrib.pop('rgba', None)
    geom.set('material', material)


def box(parent, pos, half_size, material, **attrs):
    return element(parent, 'geom', type='box', pos=vec(pos), size=vec(half_size),
                   material=material, **VISUAL, **attrs)


def add_materials(root):
    directory = ROOT/'assets/workcell'
    make_textures(directory)
    asset = root.find('asset')
    for name in ('kraft', 'birch', 'flute', 'bench_grid', 'source_liner', 'destination_liner', 'vinyl', 'ceiling_tile'):
        element(asset, 'texture', name=name+'_map', type='2d', file=str(directory/(name+'.png')))
        element(asset, 'material', name=name, texture=name+'_map', specular='.06', shininess='.12')
    recipes = [
        ('crease', '.30 .26 .21', .02, .1),
        ('nylon', '.92 .73 .285', .22, .36), ('spares', '.78 .53 .16', .15, .28),
        ('enamel', '.79 .805 .80', .23, .38), ('alloy', '.59 .625 .63', .48, .62),
        ('graphite', '.122 .136 .147', .29, .42), ('glass', '.039 .061 .073', .65, .72),
        ('visor', '.097 .133 .146', .56, .66), ('indicator', '.18 .57 .61', .2, .4),
        ('wall', '.76 .75 .72', .025, .1), ('reveal', '.45 .46 .45', .04, .1),
        ('cupboard', '.59 .625 .63', .14, .31), ('bench_edge', '.36 .32 .285', .08, .22),
        ('screen', '.065 .085 .091', .22, .55), ('frosted', '.72 .79 .80', .24, .3),
        ('steel', '.38 .41 .415', .34, .46), ('diffuser', '.93 .93 .90', .08, .2),
    ]
    for name, rgb, spec, shine in recipes:
        element(asset, 'material', name=name, rgba=rgb+' 1', specular=spec, shininess=shine)


def mesh(root, name, vertices, faces, uvs=None):
    attrs = dict(name=name, vertex=vec(np.asarray(vertices).ravel()),
                 face=' '.join(str(i) for face in faces for i in face))
    if uvs is not None:
        attrs['texcoord'] = vec(np.asarray(uvs).ravel())
    element(root.find('asset'), 'mesh', **attrs)


def top_panel(root, name, x, y, depth):
    """A thin closed panel with a full-image UV mapping on its upper face."""
    corners = [(-x,-y),(x,-y),(x,y),(-x,y)]
    verts = [(a,b,z) for z in (-depth,0.) for a,b in corners]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7)]
    for i in range(4):
        j = (i+1)%4
        faces.extend([(i,j,j+4),(i,j+4,i+4)])
    mesh(root,name,verts,faces,[(0,0),(1,0),(1,1),(0,1)]*2)


def rounded_loop(y, z, radius):
    coords = []
    for cy, cz, start in ((y-radius,z-radius,0),(-y+radius,z-radius,90),
                          (-y+radius,-z+radius,180),(y-radius,-z+radius,270)):
        for a in np.radians(np.linspace(start,start+90,6)):
            coords.append((cy+radius*np.cos(a), cz+radius*np.sin(a)))
    return coords


def loft_faces(rows, count, closed=False):
    faces = []
    for i in range(rows if closed else rows-1):
        after = (i+1)%rows
        for j in range(count):
            k = (j+1)%count
            a,b,c,d = i*count+j,i*count+k,after*count+k,after*count+j
            faces.extend([(a,b,c),(a,c,d)])
    return faces


def add_nylon(root, head, tail):
    # A single open ring shares the four collision walls' exact centre and throat.
    profile = [(-.004,.00455,.00355,.00075),(-.00385,.00484,.00384,.0009),
               (-.00345,.005,.004,.001),(.00345,.005,.004,.001),
               (.00385,.00484,.00384,.0009),(.004,.00455,.00355,.00075),
               (.004,.00303,.00173,.00036),(.00391,.00288,.00158,.00030),
               (.00347,.00250,.00120,.00012),(-.00347,.00250,.00120,.00012),
               (-.00391,.00288,.00158,.00030),(-.004,.00303,.00173,.00036)]
    vertices = [(x,y,z) for x,hy,hz,r in profile for y,z in rounded_loop(hy,hz,r)]
    mesh(root,'moulded_buckle',vertices,loft_faces(len(profile),24,closed=True))
    for name in ('top','bottom','a','b'):
        head.find(f'geom[@name="lock_{name}"]').set('group','3')
    element(head,'geom',type='mesh',mesh='moulded_buckle',quat=head.find('site[@name="hole"]').get('quat'),
            material='nylon',**VISUAL)
    for name in ('neck_bridge','grasp_neck'):
        finish(head.find(f'geom[@name="{name}"]'),'nylon')
    # Shallow moulded teeth on the broad face end before the smooth lead-in.
    sections = [(-.010,.0015,.0004,0.)]
    for x in np.arange(-.008,.034,.00085):
        sections.extend([(x,.0015,.0004,0.),(x+.00018,.0015,.0004,.000075),
                         (x+.00050,.0015,.0004,0.)])
    sections.extend([(.036,.0015,.0004,0.),(.039,.00132,.00036,0.),
                     (.0427,.00074,.00023,0.),(.044,.00019,.00014,0.)])
    verts = []
    for x,hw,hh,groove in sections:
        for y,z in ((hw*.92,-hh),(hw,-hh*.7),(hw,hh*.7),(hw*.92,hh),
                    (hw*.75,hh-groove),(-hw*.75,hh-groove),(-hw*.92,hh),
                    (-hw,hh*.7),(-hw,-hh*.7),(-hw*.92,-hh)):
            verts.append((x,y,z))
    n=10
    faces=loft_faces(len(sections),n)
    last=(len(sections)-1)*n
    for j in range(1,n-1):
        faces.extend([(0,j+1,j),(last,last+j,last+j+1)])
    mesh(root,'tapered_tongue',verts,faces)
    tail.find('geom[@name="rigid_tail"]').set('group','3')
    element(tail,'geom',type='mesh',mesh='tapered_tongue',material='nylon',**VISUAL)


def dress_trays(root, layout):
    world = root.find('worldbody')
    for label in ('source_tray','destination_tray'):
        tray=world.find(f'body[@name="{label}"]')
        x,y,z=layout[label]['half_size']
        for item in tray.findall('geom'):
            if item.get('mesh') == 'spare_ties':
                finish(item,'spares')
            elif item.get('name','').endswith('_floor'):
                finish(item,'kraft')
            else:
                finish(item,'kraft')
        name = label+'_insert'
        top_panel(root,name,x-.006,y-.006,.00016)
        element(tray,'geom',type='mesh',mesh=name,pos='0 0 .00618',
                material=label.split('_')[0]+'_liner',**VISUAL)
        for axis,span,width,angle in (('x',x,y,0),('y',y,x,90)):
            name=f'{label}_edge_{axis}'
            top_panel(root,name,span-.003,.003,.0004)
            for sign in (-1,1):
                off=sign*(width-.003)
                pos=(0,off,2*z+.00012) if axis=='x' else (off,0,2*z+.00012)
                element(tray,'geom',type='mesh',mesh=name,pos=vec(pos),
                        quat='1 0 0 0' if angle==0 else '.70710678 0 0 .70710678',
                        material='flute',**VISUAL)
        # Two scoring lines on the short folded returns, inset from the end walls.
        for sign in (-1,1):
            box(tray,(sign*(x-.018),0,.00608),(.00032,y-.006,.00006),'crease')
        for sign in (-1,1):
            box(tray,(x-.0062,sign*(y-.018),z),(.0003,.00035,z-.003),'crease')


def dress_workstation(root):
    world = root.find('worldbody')
    finish(world.find('geom[@name="floor"]'),'vinyl')
    finish(world.find('geom[@name="tabletop"]'),'birch')
    finish(world.find('geom[@name="cutting_mat"]'),'bench_grid')
    top_panel(root,'printed_grid',.31,.46,.00015)
    element(world,'geom',type='mesh',mesh='printed_grid',pos='.51 0 .75915',material='bench_grid',**VISUAL)
    # Reference room: paint, skirting, a rear worktop, cabinets and a monitor.
    for x in (-2.25,2.65):
        box(world,(x,0,1.34),(.035,2.54,1.34),'wall')
        box(world,(x+(.037 if x<0 else -.037),0,.058),(.006,2.53,.058),'reveal')
    for y in (-2.5,2.5):
        box(world,(.2,y,1.34),(2.44,.035,1.34),'wall')
        box(world,(.2,y+(.037 if y<0 else -.037),.058),(2.43,.006,.058),'reveal')
    for i in range(4):
        y=-1.4+.72*i
        box(world,(-1.84,y,.40),(.31,.353,.37),'cupboard')
        for door in (-.175,.175):
            box(world,(-1.52,y+door,.42),(.014,.17,.315),'wall')
            box(world,(-1.493,y+door,.64),(.014,.048,.008),'steel')
    box(world,(-1.86,-.32,.806),(.36,1.47,.025),'birch')
    box(world,(-1.80,-.95,.846),(.10,.22,.014),'graphite')
    box(world,(-1.86,-.95,.97),(.023,.041,.11),'steel')
    box(world,(-1.86,-.95,1.20),(.026,.34,.205),'graphite')
    box(world,(-1.832,-.95,1.205),(.002,.323,.184),'screen')
    # Acoustic ceiling and recessed light panels are visible when a wrist turns over.
    root.find('.//material[@name="ceiling_tile"]').set('texrepeat','8 8')
    root.find('.//material[@name="vinyl"]').set('texrepeat','5 5')
    root.find('.//material[@name="diffuser"]').set('emission','.27')
    box(world,(.20,0,2.70),(2.46,2.54,.025),'ceiling_tile')
    for x in (-1.0,.81):
        for y in (-.78,1.10):
            box(world,(x,y,2.669),(.326,.32,.006),'reveal')
            box(world,(x,y,2.661),(.304,.298,.003),'diffuser')
    # A few pale paper sheets rest on the worktop beside the monitor.
    for i in range(3):
        box(world,(-1.70+i*.006,-.25-i*.007,.833+i*.0012),(.12,.157,.0006),'wall')
    # Window recesses on the opposite side provide context in wrist shots.
    for y in (-1.05,0.,1.05):
        box(world,(2.608,y,1.59),(.009,.50,.53),'reveal')
        box(world,(2.593,y,1.59),(.006,.479,.504),'frosted')
        box(world,(2.581,y,1.59),(.006,.012,.51),'wall')
    box(world,(2.54,0,1.048),(.072,1.65,.017),'wall')


def add_visor(root):
    # The supplied head STL models the frame; close the visible cavity inside it.
    torso = root.find('.//body[@name="torso_link"]')
    frame = torso.find('geom[@mesh="head_link"]')
    mount = element(torso, 'body', name='visor_insert', pos=frame.get('pos'), quat=frame.get('quat'))
    element(mount, 'geom', type='ellipsoid', pos='.008 0 .431',
            size='.039 .049 .059', material='visor', **VISUAL)
    z = .446
    y = np.linspace(-.022,.022,9)
    x = .0086 + .039*np.sqrt(1-(y/.049)**2-((z-.431)/.059)**2)
    for j in range(len(y)-1):
        element(mount,'geom',type='capsule',fromto=vec([x[j],y[j],z,x[j+1],y[j+1],z]),
                size='.0008',material='indicator',**VISUAL)


def apply_appearance(root, layout, head, tail):
    add_materials(root)
    add_nylon(root,head,tail)
    add_visor(root)
    dress_trays(root,layout)
    dress_workstation(root)
    # Keep the near clipping plane and shadow volume referenced to the robot.
    element(root,'statistic',extent='1.78508',center='.34047 0 .49254')
    visual=root.find('visual')
    visual.find('quality').set('shadowsize','2048')
    visual.find('headlight').attrib.update(diffuse='.20 .205 .21',ambient='.27 .27 .275',specular='.07 .07 .07')
    visual.find('map').attrib.update(znear='.002',shadowclip='1.5',shadowscale='.65')
    key,fill=root.findall('worldbody/light')
    key.attrib.update(pos='1.2 -1.25 2.65',dir='-.64 .82 -1.62',diffuse='.59 .57 .535',
                      specular='.34 .33 .30',cutoff='57',exponent='2',attenuation='1 0 0')
    fill.attrib.update(pos='-.8 1.10 2.10',dir='.7 -.5 -.9',diffuse='.37 .39 .415',
                       specular='.17 .19 .22',cutoff='80',attenuation='1 0 0')
    element(root.find('worldbody'),'light',pos='1.6 1.1 2.5',dir='-1 -.65 -.9',
            directional='true',diffuse='.22 .22 .21',specular='.12 .12 .12',castshadow='false')
    for g in root.findall('.//worldbody//geom[@class="visual"]'):
        original=g.get('material')
        if g.get('mesh')=='head_link':
            finish(g,'glass')
        elif original!='black':
            finish(g,'alloy' if any(s in g.get('mesh','') for s in ('wrist_pitch','waist_roll','ankle_roll')) else 'enamel')
    for g in root.findall('.//worldbody//geom'):
        name=g.get('name','')
        if g.get('mesh','').startswith('Dex1_') and g.get('group')=='2':
            finish(g,'graphite' if 'finger' in g.get('mesh') else 'steel')
        elif name.endswith('_d405_case'):
            finish(g,'alloy')
        elif '_d405_lens_' in name:
            finish(g,'glass')

