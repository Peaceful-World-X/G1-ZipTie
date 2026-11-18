"""Continuous display surface driven by the physical ribbon bodies.

MuJoCo skinning changes appearance only. Contacts, mass and elastic forces
still use the underlying links. No independent animated curve is generated.
"""
import numpy as np
from build_scene import element,vec


def add_ribbon_skin(root,model,data,start=.014,end=.176):
    count=32;sections=129
    names=['lock_head']+[f'strip_{i:02}' for i in range(count)]+['tail_body']
    anchors=np.r_[start,np.linspace(start,end,count,endpoint=False)+(end-start)/count/2,end]
    vertices=[];weights=[{} for _ in names]
    head=data.body('lock_head');rotation=head.xmat.reshape(3,3)
    corners=[(-.0015,-.0004),(.0015,-.0004),(.0015,.0004),(-.0015,.0004)]
    for k,x in enumerate(np.linspace(start,end,sections)):
        j=int(np.clip(np.searchsorted(anchors,x,side='right')-1,0,len(anchors)-2))
        blend=float(np.clip((x-anchors[j])/(anchors[j+1]-anchors[j]),0,1))
        blend=blend*blend*(3-2*blend)
        for y,z in corners:
            idx=len(vertices);vertices.append(head.xpos+rotation@np.array([x,y,z-.008]))
            for b,w in ((j,1-blend),(j+1,blend)):
                if w>0:weights[b][idx]=w
    faces=[]
    for k in range(sections-1):
        for j in range(4):
            a=4*k+j;b=4*k+(j+1)%4;c=b+4;d=a+4
            faces.extend([(a,b,c),(a,c,d)])
    faces.extend([(0,2,1),(0,3,2)])
    last=4*(sections-1);faces.extend([(last,last+1,last+2),(last,last+2,last+3)])
    skin=element(root.find('asset'),'skin',name='continuous_ribbon',material='nylon',group='0',vertex=vec(np.asarray(vertices).ravel()),face=' '.join(str(i) for f in faces for i in f))
    for name,mapping in zip(names,weights):
        if not mapping:continue
        body=data.body(name)
        element(skin,'bone',body=name,bindpos=vec(body.xpos),bindquat=vec(body.xquat),vertid=' '.join(map(str,mapping)),vertweight=vec(list(mapping.values())))
    for geom in root.findall('.//geom'):
        if geom.get('name','').startswith('strip_geom_'):geom.set('group','3')
