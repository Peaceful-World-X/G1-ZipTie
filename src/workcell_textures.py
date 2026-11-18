"""Deterministic workbench surface maps. Dimensions of the mat are in centimetres."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter


def rgb_surface(color, variation):
    pixels = np.asarray(color)[None, None, :] + variation[:, :, None]
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))


def grain(rng, shape, sigma):
    field = gaussian_filter(rng.standard_normal(shape).astype('float32'), sigma, mode='wrap')
    return field / max(float(field.std()), 1e-6)


def make_textures(directory):
    rng = np.random.default_rng(405)
    shape = (512, 1024)
    paper = 3*grain(rng, shape, (.55, 2.4)) + 2*grain(rng, shape, 9) + grain(rng, shape, .25)
    rgb_surface((151, 131, 107), paper).save(directory/'kraft.png')
    wood = 3*grain(rng, shape, (1, 25)) + 4*grain(rng, shape, (4, 65)) + 2*grain(rng, shape, 1)
    rgb_surface((184, 163, 140), wood).save(directory/'birch.png')
    # Cross section of two paper liners and the fluted core at the cut edge.
    y, x = np.mgrid[0:72, 0:1536]
    ridge = 36 + 26*np.sin(2*np.pi*x/23 + .12*np.sin(x/37))
    fiber = rng.normal(0, 4, x.shape)
    shade = 52*np.exp(-((y-ridge)/4.8)**2) + 31*((y < 6) | (y > 65)) + fiber
    rgb_surface((91, 72, 51), shade).save(directory/'flute.png')
    # Occlusion at the four internal folds fades over about a centimetre.
    v,u = np.mgrid[0:400,0:640]
    edge = np.minimum(np.minimum(u,639-u), np.minimum(v,399-v))
    dark = -36*np.exp(-edge/13) - 3*np.exp(-edge/44)
    fibers = 2.5*grain(rng, edge.shape,(.6,2.6)) + 2*grain(rng, edge.shape,12)
    for tray,color in (('source', (173,162,148)), ('destination', (153,130,104))):
        liner = rgb_surface(color, fibers+dark)
        crease = ImageDraw.Draw(liner)
        for x0,sign in ((0,1),(639,-1)):
            for y0,direction in ((0,1),(399,-1)):
                crease.line((x0,y0,x0+sign*28,y0+direction*29),fill=(117,100,80))
        liner.save(directory/(tray+'_liner.png'))
    flecks = 3*grain(rng,shape,.55) + 2.5*grain(rng,shape,3)
    rgb_surface((152,155,151),flecks).save(directory/'vinyl.png')
    ceiling = rgb_surface((208,207,200), grain(rng,(256,256),.8)*2)
    edge_draw = ImageDraw.Draw(ceiling)
    edge_draw.rectangle((0,0,255,255),outline=(140,144,142),width=2)
    edge_draw.rectangle((3,3,252,252),outline=(190,191,184),width=1)
    ceiling.save(directory/'ceiling_tile.png')
    # Printed grids and rulers lie on the same UV-mapped surface as the knife marks.
    width, height, unit = 1240, 1840, 20
    grain_map = grain(rng, (height, width), .5)*1.8
    mat = rgb_surface((29, 102, 110), grain_map)
    draw = ImageDraw.Draw(mat)
    for pos in range(20, width-19, unit//2):
        color = (98, 150, 150) if (pos-20)%unit == 0 else (56, 125, 131)
        draw.line((pos,20,pos,height-20), fill=color, width=2 if (pos-20)%(5*unit)==0 else 1)
    for pos in range(20, height-19, unit//2):
        color = (98, 150, 150) if (pos-20)%unit == 0 else (56, 125, 131)
        draw.line((20,pos,width-20,pos), fill=color, width=2 if (pos-20)%(5*unit)==0 else 1)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    for cm in range(5, 60, 5):
        draw.text((20+cm*unit+2,3),str(cm),fill=(157,183,175),font=font)
    for cm in range(5, 90, 5):
        draw.text((3,20+cm*unit),str(cm),fill=(157,183,175),font=font)
    for side in range(2, 10):
        size = unit*side
        draw.line([(width-260-size,height-145), (width-260,height-145-size),
                   (width-260+size,height-145)],fill=(107,154,151),width=2)
    for _ in range(70):
        x, y = rng.uniform((55,75),(width-90,height-80))
        draw.line((x,y,x+rng.uniform(-95,95),y+rng.uniform(-75,75)), fill=(63,121,124),width=1)
    mat.save(directory/'bench_grid.png')

