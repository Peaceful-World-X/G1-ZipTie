"""Render a shareable Chinese presentation video and measured trajectory."""
import argparse
import json
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from simulation import *
from replay import G1ZipTieEnv,DURATION,PHASES,SCRIPT_RATE
from timeline import Timeline

def font(size,bold=False):
    path=Path('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc')
    if not path.exists(): path=Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path),size)

F={n:font(n,n>=26) for n in (15,18,20,24,26,30,44)}

def compose(rgb,env,view,feeds=None):
    im=Image.new('RGB',(1600,1000),'#0b131d'); d=ImageDraw.Draw(im)
    d.rounded_rectangle((26,27,66,67),10,fill='#a9ef62')
    d.text((34,31),'G1',font=F[20],fill='#12201c')
    d.text((82,27),'ZIP-TIE LAB',font=F[30],fill='#edf4f4')
    d.text((82,67),'宇树 G1 × Dex1-1  /  弹性扎带穿扣仿真',font=F[18],fill='#97a9b7')
    d.text((1210,34),'LOCAL SIMULATION',font=F[20],fill='#a9ef62')
    d.text((1210,66),'MuJoCo 3.3.6 · Windows',font=F[15],fill='#97a9b7')
    im.paste(Image.fromarray(rgb).resize((1120,740)),(24,120))
    d=ImageDraw.Draw(im)
    d.rounded_rectangle((44,140,245,177),8,fill='#101e2b')
    d.text((58,147),{'overview':'01 / 工作站全景','detail':'02 / 穿扣近景','slot':'锁扣侧面近景','hands':'02 / 双手操作','top':'03 / 顶部视角'}[view],font=F[18],fill='#d9e7e9')
    for i,(label,feed) in enumerate(zip(('左手腕 / D405','右手腕 / D405','机器人前方俯视 / FRONT DOWN'),feeds or [])):
        y=120+i*248
        picture = Image.fromarray(feed)
        if i < 2:
            width, height = wrist_frame_size(*picture.size)
            x0, y0 = (picture.width-width)//2, (picture.height-height)//2
            picture = picture.crop((x0, y0, x0+width, y0+height))
            picture = picture.resize((328, 246), Image.Resampling.LANCZOS)
            x = 1166 + (410-picture.width)//2
            im.paste(picture, (x,y))
            d.rectangle((x,y,x+151,y+27),fill='#162a38')
            d.text((x+8,y+2),label,font=F[18],fill='#d8edf0')
        else:
            d.rectangle((1166,y,1576,y+32),fill='#162a38')
            d.text((1180,y+5),label,font=F[18],fill='#d8edf0')
            im.paste(picture.resize((410,214), Image.Resampling.LANCZOS),(1166,y+32))
    d=ImageDraw.Draw(im)
    d.rounded_rectangle((44,190,350,235),8,fill='#142934')
    d.text((59,197),env.phase()[2],font=F[24],fill='#b7f47a')
    info=env.info()
    d.rounded_rectangle((44,783,1124,842),10,fill='#152633')
    for i,(label,value) in enumerate([('穿入深度',f'{info["depth_mm"]:.1f} mm'),('横向偏差',f'{info["lateral_mm"]:.2f} mm'),('轴线夹角',f'{info["angle_deg"]:.2f}°')]):
        x=62+i*350
        d.text((x,800),label,font=F[18],fill='#8ea5b5')
        d.text((x+115,796),value,font=F[24],fill='#e9f3f4')
    good=info['inserted']
    if good:
        d.rounded_rectangle((774,140,1124,182),9,fill='#254e3c')
        d.text((790,149),'任务完成 / 已放入盒中' if info['task_complete'] else '已验证穿入',font=F[20],fill='#b9f58e')
    d.rounded_rectangle((24,883,1576,891),4,fill='#263747')
    if env.data.time>0:d.rounded_rectangle((24,883,24+1552*min(env.data.time/DURATION,1),891),4,fill='#a9ef62')
    d.text((24,910),'3 mm × 0.8 mm 细带 · 32 段弹性动力学 · 官方 Dex1-1 模型',font=F[20],fill='#d8e3e9')
    d.text((24,947),'右手持带尾，左手持锁扣；弹性参数未标定，夹持与穿入后的锁定采用简化约束。',font=F[18],fill='#849ba9')
    d.text((1360,916),f'{env.data.time:05.2f} / {DURATION:.0f} s',font=F[20],fill='#b0c3ce')
    return np.array(im)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fps',type=int,default=30); parser.add_argument('--preview',action='store_true')
    parser.add_argument('--live',action='store_true',help='Recompute physics instead of rendering the verified cache.')
    args=parser.parse_args()
    out=ROOT/'recordings'; out.mkdir(exist_ok=True)
    env=G1ZipTieEnv(); trace=[]
    timeline=None if args.live else Timeline(env)
    writer=None if args.preview else imageio.get_writer(str(out/'G1_穿扎带_演示.mp4'),fps=args.fps,codec='libx264',quality=8,macro_block_size=2)
    try:
        count=1 if args.preview else int(DURATION*args.fps)
        for i in range(count):
            t=18 if args.preview else i/args.fps
            if timeline:timeline.display(t)
            else:
                while env.data.time<t-1e-8: env.step(dt=env.model.opt.timestep)
            view='overview' if t<3 or t>=19 else ('hands' if t<12.6 else 'slot')
            rgb=render(env,view,width=1120,height=740)
            feeds=[render_sensor(env,name,width=656,height=344) for name in ('left_wrist_cam','right_wrist_cam','front_down_cam')]
            frame=compose(rgb,env,view,feeds)
            if writer: writer.append_data(frame)
            if args.preview or i in (0,18*args.fps):
                Image.fromarray(frame).save(out/('preview.png' if args.preview or i else 'overview.png'))
            if not args.preview and i in [s*args.fps for s in (3,5,7,8,9,10,12,16,18,23)]:
                Image.fromarray(frame).save(out/f'stage_{int(t):02}.png')
            trace.append({k:v for k,v in env.info().items() if k!='local_tip'})
            if i%(args.fps*3)==0: print(f'Rendering {i}/{count} frames, t={env.data.time:.2f}s',flush=True)
        report={**env.info(),'mujoco_version':mujoco.__version__,'frames':count,'fps':args.fps,'source':'verified physics state cache' if timeline else 'live solver','warnings':int(env.data.warning.number.sum())}
        (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with (out/'trajectory.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=trace[0].keys()); w.writeheader(); w.writerows(trace)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    finally:
        if writer: writer.close()
        env.close()

if __name__=='__main__': main()
