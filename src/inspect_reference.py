from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import numpy as np
source=Path('D:/Desktop/20261113_145314_785964127_561906_g1.mp4')
reader=imageio.get_reader(str(source)); meta=reader.get_meta_data()
times=np.linspace(0,max(0,meta['duration']-.2),16)
sheet=Image.new('RGB',(1280,4*220),'#17202a'); draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
out=Path(__file__).resolve().parents[1]/'recordings'
for i,t in enumerate(times):
    frame=Image.fromarray(reader.get_data(int(t*meta['fps']))); frame.thumbnail((316,188))
    x=(i%4)*320; y=(i//4)*220
    sheet.paste(frame,(x,y)); draw.text((x+8,y+190),f'{t:.1f} s',fill='white',font=font)
sheet.save(out/'reference_contact_sheet.jpg'); reader.close()
