"""Native interactive MuJoCo presentation. Run with the local virtualenv."""
import time
import argparse
import json
from collections import deque
import glfw
import mujoco
import numpy as np
from OpenGL.GL import glGetString,GL_RENDERER,glClear,glClearColor,glDisable,GL_COLOR_BUFFER_BIT,GL_DEPTH_BUFFER_BIT,GL_SCISSOR_TEST
from PIL import Image
from simulation import ROOT, camera, add_loop, wrist_frame_size
from replay import G1ZipTieEnv, DURATION, PHASES
from timeline import Timeline

BAR_HEIGHT=96

def progress_geometry(width):
    return 132, max(133,width-36), 34

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=0,help='Auto-close after N seconds (smoke test).')
    parser.add_argument('--paused',action='store_true',help='Open at the initial tray state; Space starts playback.')
    parser.add_argument('--start-at',type=float,default=0,help='Open the timeline at this second.')
    parser.add_argument('--fps',type=int,default=30,choices=(20,25,30),help='Presentation refresh rate.')
    parser.add_argument('--live',action='store_true',help='Run the physics solver live instead of cached presentation.')
    parser.add_argument('--profile',action='store_true',help='Save measured window frame intervals on exit.')
    args=parser.parse_args()
    env=G1ZipTieEnv()
    if not glfw.init(): raise RuntimeError('GLFW initialization failed')
    glfw.window_hint(glfw.SAMPLES,4)
    window=glfw.create_window(1600,1000,'G1 ZIP-TIE LAB | D405 | 24s | Drag timeline to seek | 3 live cameras',None,None)
    if not window: raise RuntimeError('OpenGL window creation failed')
    glfw.make_context_current(window); glfw.swap_interval(0)
    scene=mujoco.MjvScene(env.model,maxgeom=5000)
    context=mujoco.MjrContext(env.model,mujoco.mjtFontScale.mjFONTSCALE_150)
    opt=mujoco.MjvOption(); opt.geomgroup[3]=0; opt.sitegroup[:]=0
    def loading(fraction):
        if glfw.window_should_close(window): raise InterruptedError('Playback preparation cancelled')
        w,h=glfw.get_framebuffer_size(window)
        glClearColor(.025,.045,.065,1);glClear(GL_COLOR_BUFFER_BIT)
        rect=mujoco.MjrRect(0,0,w,h)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,rect,
                           f'Preparing playback timeline: {fraction:.0%}\nThis is cached for future launches.','',context)
        mujoco.mjr_rectangle(mujoco.MjrRect(24,h//2,int((w-48)*fraction),10),.65,.94,.38,1)
        glfw.swap_buffers(window);glfw.poll_events()
    try:
        timeline=Timeline(env,progress=loading)
    except BaseException:
        env.close();context.free();glfw.destroy_window(window);glfw.terminate();raise
    start_at=float(np.clip(args.start_at,0,DURATION))
    if not np.isfinite(start_at):parser.error('--start-at must be finite')
    cam=camera('slot' if 12.6<=start_at<19 else 'hands' if 3<=start_at<12.6 else 'overview')
    state={'paused':args.paused,'loop':True,'speed':1.,'last':None,'screenshot':False,
           'dragging':False,'resume':False,'seek':start_at,'target':start_at,'cached':not args.live}
    sensor_cameras=[]
    for name,label in [('left_wrist_cam','LEFT WRIST / D405'),('right_wrist_cam','RIGHT WRIST / D405'),('front_down_cam','ROBOT FRONT / LOOKING DOWN')]:
        fixed=mujoco.MjvCamera();fixed.type=mujoco.mjtCamera.mjCAMERA_FIXED;fixed.fixedcamid=env.model.camera(name).id
        sensor_cameras.append((fixed,label))
    def queue_seek(seconds):
        state['target']=float(np.clip(seconds,0,DURATION));state['seek']=state['target']
    def pointer(win,x=None,y=None):
        if x is None:x,y=glfw.get_cursor_pos(win)
        ww,wh=glfw.get_window_size(win);fw,fh=glfw.get_framebuffer_size(win)
        return x*fw/max(ww,1),fh-y*fh/max(wh,1)
    def pointer_time(win,x):
        left,right,_=progress_geometry(glfw.get_framebuffer_size(win)[0])
        return DURATION*(x-left)/(right-left)
    def key(win,k,scan,action,mods):
        nonlocal cam
        if action!=glfw.PRESS: return
        if k==glfw.KEY_ESCAPE: glfw.set_window_should_close(win,True)
        elif k==glfw.KEY_SPACE and not state['dragging']: state['paused']=not state['paused']
        elif k==glfw.KEY_R: queue_seek(0); state['paused']=False
        elif k==glfw.KEY_LEFT: queue_seek(env.data.time-1)
        elif k==glfw.KEY_RIGHT: queue_seek(env.data.time+1)
        elif k==glfw.KEY_HOME: queue_seek(0)
        elif k==glfw.KEY_END: queue_seek(DURATION);state['paused']=True
        elif k in (glfw.KEY_1,glfw.KEY_2,glfw.KEY_3,glfw.KEY_4): cam=camera({glfw.KEY_1:'overview',glfw.KEY_2:'detail',glfw.KEY_3:'top',glfw.KEY_4:'slot'}[k])
        elif k==glfw.KEY_L: state['loop']=not state['loop']
        elif k==glfw.KEY_S: state['screenshot']=True
        elif k==glfw.KEY_P:
            if state['cached']:timeline.seek(env.data.time)
            else:timeline.display(env.data.time)
            state['cached']=not state['cached']
        elif k==glfw.KEY_B:
            if state['cached']:timeline.seek(env.data.time);state['cached']=False
            env.perturb_strip()
        elif k==glfw.KEY_MINUS: state['speed']=max(.25,state['speed']/2)
        elif k==glfw.KEY_EQUAL: state['speed']=min(2.,state['speed']*2)
    def cursor(win,x,y):
        fx,fy=pointer(win,x,y)
        if state['dragging']:
            queue_seek(pointer_time(win,fx));state['last']=(x,y);return
        if state['last'] is not None:
            dx=x-state['last'][0]; dy=y-state['last'][1]
            h=glfw.get_window_size(win)[1]
            in_main=fx<glfw.get_framebuffer_size(win)[0]*.72 and fy>BAR_HEIGHT
            if in_main and glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
                mujoco.mjv_moveCamera(env.model,mujoco.mjtMouse.mjMOUSE_ROTATE_V,dx/h,dy/h,scene,cam)
            elif in_main and glfw.get_mouse_button(win,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS:
                mujoco.mjv_moveCamera(env.model,mujoco.mjtMouse.mjMOUSE_MOVE_V,dx/h,dy/h,scene,cam)
        state['last']=(x,y)
    def scroll(win,x,y):
        fx,fy=pointer(win)
        if fx<glfw.get_framebuffer_size(win)[0]*.72 and fy>BAR_HEIGHT:
            mujoco.mjv_moveCamera(env.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.06*y,scene,cam)
    def mouse(win,button,action,mods):
        if button!=glfw.MOUSE_BUTTON_LEFT:return
        fx,fy=pointer(win)
        if action==glfw.PRESS and 12<=fy<=72:
            if 18<=fx<=108:
                state['paused']=not state['paused']
            elif fx>=120:
                state['resume']=not state['paused'];state['paused']=True;state['dragging']=True
                queue_seek(pointer_time(win,fx))
        elif action==glfw.RELEASE and state['dragging']:
            state['dragging']=False;queue_seek(pointer_time(win,fx));state['paused']=not state['resume']
    def focus(win,focused):
        if not focused and state['dragging']:
            state['dragging']=False;state['paused']=True;queue_seek(state['target'])
    glfw.set_key_callback(window,key); glfw.set_cursor_pos_callback(window,cursor); glfw.set_scroll_callback(window,scroll)
    glfw.set_mouse_button_callback(window,mouse);glfw.set_window_focus_callback(window,focus)
    start=last=time.perf_counter(); accum=0.;deadline=start
    frame_times=deque(maxlen=61);frame_samples=[];previous_present=None;fps=0.
    try:
        while not glfw.window_should_close(window):
            now=time.perf_counter(); elapsed=min(now-last,.1); last=now
            if args.seconds and now-start>args.seconds: break
            if state['seek'] is not None:
                if state['cached']:timeline.display(state['seek'])
                else:timeline.seek(state['seek'],preview=state['dragging'])
                state['seek']=None;accum=0.;elapsed=0.;last=time.perf_counter()
            if not state['paused']:
                if state['cached']:
                    target=env.data.time+elapsed*state['speed']
                    if target>DURATION and state['loop']:target%=DURATION
                    timeline.display(min(target,DURATION))
                else:
                    accum+=elapsed*state['speed']
                    while accum>=.01:
                        if env.data.time<DURATION: env.step(dt=.01)
                        elif state['loop']: timeline.seek(0)
                        accum-=.01
            w,h=glfw.get_framebuffer_size(window)
            if w<200 or h<=BAR_HEIGHT:
                glfw.poll_events();continue
            main_width=int(w*.72)
            viewport=mujoco.MjrRect(0,BAR_HEIGHT,main_width,h-BAR_HEIGHT)
            full_viewport=mujoco.MjrRect(0,0,w,h)
            glDisable(GL_SCISSOR_TEST);glClearColor(.025,.045,.065,1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            mujoco.mjv_updateScene(env.model,env.data,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scene)
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=1
            add_loop(scene,env)
            mujoco.mjr_render(viewport,scene,context)
            info=env.info(); status='INSERTION VERIFIED' if info['success'] else env.phase()[1]
            if info['task_complete']: status='PLACED IN TRAY / COMPLETE'
            if info['failure']: status='FAILED / '+info['failure']
            if state['paused']:status+=' / PAUSED - SPACE TO PLAY'
            title='G1 / DEX1-1 / D405\n'+status
            mode='CACHED PHYSICS' if state['cached'] else 'LIVE SOLVER'
            values=f'Time   {env.data.time:05.2f} / {DURATION:.2f} s\nDepth  {info["depth_mm"]:6.2f} mm\nOffset {info["lateral_mm"]:6.2f} mm\nAngle  {info["angle_deg"]:6.2f} deg\nHold   {info["hold_seconds"]:6.2f} s\nFPS    {fps:4.1f} / {args.fps}\n{mode}'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,title,'',context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPRIGHT,viewport,values,'',context)
            controls='SPACE pause | R reset | 1/2/3 cameras | 4 slot | P cached/live | B live perturb\nS snapshot | +/- speed | L loop | Arrow keys seek 1s | ESC exit'
            scope='Official Dex1-1 / 3mm x 0.8mm elastic strip / 32 physical segments / fixed base / no ratchet'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,controls+'\n'+scope,'',context)
            for i,(fixed,label) in enumerate(sensor_cameras):
                row_h=(h-BAR_HEIGHT)//3
                sub=mujoco.MjrRect(main_width+4,h-(i+1)*row_h,w-main_width-4,row_h-4)
                if i < 2:
                    width, height = wrist_frame_size(sub.width, sub.height)
                    sub = mujoco.MjrRect(sub.left + (sub.width-width)//2, sub.bottom + (sub.height-height)//2,
                                         width, height)
                # The four views share the skinned geometry at this timestamp.
                mujoco.mjv_updateCamera(env.model,env.data,fixed,scene)
                mujoco.mjv_makeLights(env.model,env.data,scene)
                mujoco.mjr_render(sub,scene,context)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,sub,label,'',context)
            # A full-width seek bar, separate from the orbitable 3-D viewport.
            mujoco.mjr_rectangle(mujoco.MjrRect(0,0,w,BAR_HEIGHT),.035,.065,.09,1)
            mujoco.mjr_rectangle(mujoco.MjrRect(18,16,90,50),.12,.22,.27,1)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               mujoco.MjrRect(22,16,86,43),'PLAY' if state['paused'] else 'PAUSE','',context)
            left,right,y=progress_geometry(w)
            shown=state['target'] if state['dragging'] else env.data.time
            thumb=left+int((right-left)*np.clip(shown/DURATION,0,1))
            mujoco.mjr_rectangle(mujoco.MjrRect(left,y-4,right-left,8),.16,.25,.29,1)
            mujoco.mjr_rectangle(mujoco.MjrRect(left,y-4,max(1,thumb-left),8),.65,.94,.38,1)
            for marker,_,_ in PHASES[1:]:
                mx=left+int((right-left)*marker/DURATION)
                mujoco.mjr_rectangle(mujoco.MjrRect(mx,y-7,2,14),.48,.58,.60,1)
            mujoco.mjr_rectangle(mujoco.MjrRect(thumb-5,y-10,10,20),.85,1.,.7,1)
            caption=f'{shown:05.2f} / {DURATION:.2f} s  |  Drag to seek  |  {state["speed"]:.2f}x  |  {fps:.1f} FPS  |  {mode}'
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               mujoco.MjrRect(left,50,right-left,40),caption,'',context)
            if state['screenshot']:
                pixels=np.empty((h,w,3),dtype=np.uint8)
                mujoco.mjr_readPixels(pixels,None,full_viewport,context)
                (ROOT/'recordings').mkdir(exist_ok=True)
                path=ROOT/'recordings'/f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.png'
                Image.fromarray(np.flipud(pixels)).save(path)
                print(path,flush=True); state['screenshot']=False
            glfw.swap_buffers(window); glfw.poll_events()
            presented=time.perf_counter();frame_times.append(presented)
            if len(frame_times)>1:fps=(len(frame_times)-1)/(frame_times[-1]-frame_times[0])
            if previous_present is not None and state['cached'] and not state['paused'] and presented-start>1:
                frame_samples.append(presented-previous_present)
            previous_present=presented
            deadline+=1/args.fps
            remaining=deadline-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            elif remaining<-1/args.fps:deadline=time.perf_counter()
    finally:
        if args.profile and frame_samples:
            durations=np.array(frame_samples)
            report={'opengl_renderer':glGetString(GL_RENDERER).decode(), 'target_fps':args.fps,'mode':'cached verified physics','sample_frames':len(durations),'mean_fps':float(1/durations.mean()),'p95_frame_ms':float(np.percentile(durations,95)*1000),'max_frame_ms':float(durations.max()*1000),'frames_below_20fps':int((durations>.05).sum()),'resolution':list(glfw.get_framebuffer_size(window))}
            (ROOT/'recordings/window_performance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(report),flush=True)
        env.close(); context.free(); glfw.destroy_window(window); glfw.terminate()

if __name__=='__main__': main()
