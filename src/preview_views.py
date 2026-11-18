"""Export the initial head/wrist comparison in the user's reference arrangement."""
from PIL import Image, ImageDraw
from replay import G1ZipTieEnv
from simulation import ROOT, render_sensor
from record_demo import font

e=G1ZipTieEnv()
try:
    im=Image.new('RGB',(1280,1100),'#0b131d');d=ImageDraw.Draw(im)
    for name,label,pos,size in [
        ('front_down_cam','HEAD / 机器人前方俯视',(240,50),(800,600)),
        ('left_wrist_cam','LEFT WRIST / 左手腕',(24,700),(600,450)),
        ('right_wrist_cam','RIGHT WRIST / 右手腕',(656,700),(600,450)),
    ]:
        # Use one 4:3 render resolution across all feeds.
        rgb=render_sensor(e,name,800,600)
        size=(size[0],min(size[1],1100-pos[1]))
        if name!='front_down_cam':size=(528,396)
        im.paste(Image.fromarray(rgb).resize(size),pos)
        d.text((pos[0],pos[1]-38),label,font=font(24),fill='#cde2e5')
    im.save(ROOT/'recordings/initial_three_views.png')
finally:
    e.close()
