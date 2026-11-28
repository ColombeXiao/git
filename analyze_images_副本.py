"""Extract wall ridges and metal scattering peaks from the reconstructed grids."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def store_csv(file, rows):
    with file.open('w', encoding='utf-8-sig', newline='') as out:
        w = csv.DictWriter(out,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def refine_max(row, grid, index):
    if index < 1 or index >= len(grid)-1:
        return float(grid[index])
    a,b,c = row[index-1:index+2]
    u = .5*(a-c)/(a+c-2*b) if a+c-2*b<0 else 0.0
    return float(grid[index]+u*(grid[index+1]-grid[index]))


def trace_ridge(a,y,limits):
    sub = (y>=limits[0]) & (y<=limits[1])
    idx = np.flatnonzero(sub)[np.argmax(a[:,sub],axis=1)]
    return np.array([refine_max(row,y,i) for row,i in zip(a,idx)]),a[np.arange(len(a)),idx]


def wall_ridges(x,y,image):
    f, amp = trace_ridge(image,y,(25,85))
    strongest = amp >= .68*np.max(amp)
    center = int(np.argmax(amp))
    left,right = center,center
    while left>0 and strongest[left-1]:
        left-=1
    while right<len(x)-1 and strongest[right+1]:
        right+=1
    fit_a = (x>=x[left]) & (x<=x[right])
    a = np.polyfit(x[fit_a],f[fit_a],1,w=np.sqrt(amp[fit_a]))
    flanks = ((x < x[left]-8) | (x > x[right]+8)) & (x>=x[0]+3) & (x<=x[-1]-3)
    outer_y = np.median(image[flanks],axis=0)
    deep = (y>=180)&(y<=230)
    j = np.flatnonzero(deep)[np.argmax(outer_y[deep])]
    rear_mode = refine_max(outer_y,y,j)
    b_track, _ = trace_ridge(image,y,(rear_mode-10,rear_mode+10))
    b = np.polyfit(x[flanks],b_track[flanks],1)
    return {'a_slope':a[0],'a_intercept_cm':a[1],'a_angle_deg':np.degrees(np.arctan(a[0])),
            'a_supported_from_x_cm':x[left],'a_supported_to_x_cm':x[right],
            'a_y_at_x25_cm':np.polyval(a,25),'a_y_at_x106_cm':np.polyval(a,106),
            'b_slope':b[0],'b_intercept_cm':b[1],'b_mode_cm':rear_mode,'b_y_at_x60_cm':np.polyval(b,60),
            'b_angle_deg':np.degrees(np.arctan(b[0]))}, f, b_track, fit_a, flanks


def brightness_at(im,y,depths):
    return np.array([np.interp(d,y,row) for d,row in zip(depths,im)])


def wall_visibility(x,y,raw,enhanced,m,fit_a,flanks):
    front = m['a_slope']*x+m['a_intercept_cm']
    rear = m['b_slope']*x+m['b_intercept_cm']
    middle = (y>=130)&(y<=178)
    rows = []
    for label,I in (('before',raw),('after',enhanced)):
        fa = np.median(brightness_at(I,y,front)[fit_a])
        ra = np.median(brightness_at(I,y,rear)[flanks])
        background = np.median(I[flanks][:,middle].max(axis=1))
        local = np.median(np.concatenate([brightness_at(I,y,rear+k)[flanks] for k in (-20,-14,14,20)]))
        rows.append({'image':label,'front_median':fa,'rear_flank_median':ra,'middle_peak_median':background,
                     'rear_vs_front_db':20*np.log10(ra/fa),'rear_vs_midband_db':20*np.log10(ra/background),
                     'rear_vs_local_background_db':20*np.log10(ra/local)})
    return rows


def estimate_objects(z,x,y,volume,y_max):
    X,Y = np.meshgrid(x,y,indexing='ij')
    region = (Y>=30) & (Y<=y_max)
    field = np.where(region[None], volume, -np.inf)
    candidates = []
    for _ in range(2):
        k,i,j = np.unravel_index(np.argmax(field),field.shape)
        point = np.array([x[i],y[j],z[k]])
        local = (np.abs(X-x[i])<=22) & (np.abs(Y-y[j])<=22)
        vmax = volume[k,i,j]
        support = local[None] & (volume>=.8*vmax)
        candidates.append({
            'x_cm':point[0],'y_cm':point[1],'z_cm':point[2],'image_value':vmax,
            'z_80pct_min_cm': min(z[np.any(support,axis=(1,2))]),
            'z_80pct_max_cm': max(z[np.any(support,axis=(1,2))]),
        })
        field[:,(X-x[i])**2+(Y-y[j])**2<=45**2] = -np.inf
    candidates.sort(key=lambda row:row['x_cm'])
    for row,label in zip(candidates,('pole','shovel')):
        row['target'] = label
    return candidates


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--results',type=Path,default=HERE/'results')
    d=cli.parse_args().results
    w=np.load(d/'wall_2d.npz')
    m,f,b,fa,fb=wall_ridges(w['x'],w['y'],w['hybrid'])
    variants=[{'scene':'walls','image':'row4_xy',**m}]
    store_csv(d/'wall_2d_ridges.csv',[
        {'x_cm':xx,'raw_front_ridge_y_cm':yy,'raw_rear_ridge_y_cm':rr,'front_fit_used':int(af),
         'rear_fit_used':int(bf)} for xx,yy,rr,af,bf in zip(w['x'],f,b,fa,fb)])
    v=wall_visibility(w['x'],w['y'],w['hybrid'],w['enhanced'],m,fa,fb)
    store_csv(d/'rear_wall_visibility.csv',v)
    walls=np.load(d/'walls_3d.npz')
    attrs = ('hybrid','envelope','even_positions','odd_positions')
    for key in attrs:
        t,*_=wall_ridges(walls['x'],walls['y'],np.median(walls[key],axis=0))
        variants += [{'scene':'walls','image':key,**t}]
    store_csv(d/'wall_positions.csv',variants)
    wall3=variants[1]
    volume=walls['hybrid']
    a_y=wall3['a_slope']*walls['x']+wall3['a_intercept_cm']
    b_y=wall3['b_slope']*walls['x']+wall3['b_intercept_cm']
    am=(walls['x']>=wall3['a_supported_from_x_cm']) & (walls['x']<=wall3['a_supported_to_x_cm'])
    bm=(walls['x']<wall3['a_supported_from_x_cm']-8) | (walls['x']>wall3['a_supported_to_x_cm']+8)
    vertical=[]
    for z,val in zip(walls['z'],volume):
        vertical.append({'z_cm':z,'front_ridge_median':np.median(brightness_at(val,walls['y'],a_y)[am]),
                         'rear_flanks_median':np.median(brightness_at(val,walls['y'],b_y)[bm])})
    store_csv(d/'wall_height_support.csv',vertical)
    metal=np.load(d/'metal_3d.npz')
    # A 20 cm clearance from the wall plane separates compact foreground
    # objects from the extended room boundary. No x or z locations are supplied.
    y_max=wall3['b_y_at_x60_cm']-20
    objects=[]
    for key in attrs:
        for p in estimate_objects(metal['z'],metal['x'],metal['y'],metal[key],y_max):
            objects.append({'image':key,**p})
    store_csv(d/'metal_positions.csv',objects)
    details={'2d':variants[0],'3d':wall3,'visibility':v,
             'metal':[x for x in objects if x['image']=='hybrid'],'foreground_y_max':y_max}
    (d/'localization_summary.json').write_text(json.dumps(details,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps(details,indent=2,default=lambda x:x.item()))


if __name__=='__main__':
    main()
