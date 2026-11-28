"""重算优化稿新增的诊断统计和图表。使用已保存的实测结果，不改变求解参数。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_images import wall_ridges, wall_visibility

HERE = Path(__file__).resolve().parent
SYSTEM_FONT = Path('/System/Library/Fonts/STHeiti Light.ttc')


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def diagnostics(results, calibration):
    delays = read_rows(calibration / 'channel_delays.csv')
    bands = Counter(r['status'] for r in read_rows(calibration / 'frequency_correction.csv'))
    assert len(delays) == 63 and sum(bands.values()) == 63 * 1401
    signal = read_rows(calibration / 'signal_audit.csv')
    spreads = [float(r['median_period_rms_difference_over_signal']) for r in signal]
    held = [r for r in read_rows(results / 'ranges_by_group.csv') if r['split'] == 'holdout']
    leave_one = {}
    for material in ('wall', 'metal', 'combined'):
        errors = np.array([abs(float(r['selected_error_cm'])) for r in held
                           if material in (r['material'], 'combined')])
        scores = (errors.sum() - errors)/(len(errors) - 1)
        leave_one[material] = {'n': len(errors), 'minimum_mae_cm': float(scores.min()),
                               'maximum_mae_cm': float(scores.max())}
    with np.load(results / 'wall_2d.npz') as raw:
        x, y, img = raw['x'], raw['y'], raw['hybrid']
        wall, _, _, front, rear = wall_ridges(x, y, img)
        scenarios = []
        for gamma in (0.0, 0.8, 1.2, 1.6):
            gain = np.minimum(3, np.maximum(1, y/80.0)**gamma)
            _, vis = wall_visibility(x, y, img, img*gain, wall, front, rear)
            scenarios.append({'exponent': gamma, 'rear_vs_front_db': vis['rear_vs_front_db'],
                              'rear_vs_local_background_db': vis['rear_vs_local_background_db']})
        assert np.allclose(np.minimum(3, np.maximum(1, y/80.0)**1.2), raw['gain'])
    return {
        'frequency_status_counts': dict(bands),
        'repeat_error_median': float(np.median(spreads)),
        'flat_matched_packets': int(sum(int(r['matched_packets']) for r in signal)),
        'leave_one_holdout_mae_cm': leave_one,
        'fixed_image_gain_sensitivity': scenarios,
        'clock_sensitivity': read_rows(results/'range_origin_sensitivity.csv'),
    }


def flow_diagram(destination, font):
    im = Image.new('RGB', (1600, 580), 'white')
    d = ImageDraw.Draw(im)
    title, regular = font(42), font(32)
    nodes = [
        (42, 34, '读取与同步', '八行共同包'),
        (595, 34, '通道校正', '扣除几何时差'),
        (1148, 34, '空场对齐', '抑制直射波'),
        (1148, 335, '垂直测距', '留出组评价'),
        (595, 335, '二维墙体', '补偿深度衰减'),
        (42, 335, '三维成像', '墙面与金属目标'),
    ]
    palette = [('#eaf3fb', '#32628a'), ('#eaf3fb', '#32628a'), ('#eaf3fb', '#32628a'),
               ('#fff2e5', '#935618'), ('#e9f4ee', '#286c53'), ('#e9f4ee', '#286c53')]
    for (x, y, top, bottom), (fill, stroke) in zip(nodes, palette):
        d.rounded_rectangle((x,y,x+410,y+210), radius=25, fill=fill, outline=stroke, width=4)
        d.text((x+205,y+73),top,font=title,fill='#172e41',anchor='mm')
        d.text((x+205,y+143),bottom,font=regular,fill='#304b59',anchor='mm')
    def arrow(x1, y1, x2, y2):
        color = '#475b6b'
        d.line((x1,y1,x2,y2),fill=color,width=6)
        xx, yy = np.array([x2-x1, y2-y1])/np.hypot(x2-x1,y2-y1)
        d.polygon([(x2,y2),(x2-24*xx+11*yy,y2-24*yy-11*xx),
                   (x2-24*xx-11*yy,y2-24*yy+11*xx)],fill=color)
    arrow(452,139,595,139)
    arrow(1005,139,1148,139)
    arrow(1353,244,1353,335)
    d.line((247,289,1353,289),fill='#475b6b',width=6)
    arrow(800,289,800,335)
    arrow(247,289,247,335)
    im.save(destination, dpi=(400,400), optimize=True)


def clock_plot(destination, summary, font):
    im = Image.new('RGB', (1580, 770), 'white')
    d = ImageDraw.Draw(im)
    left, right, top, bottom = 186, 1496, 75, 595
    label, tick = font(37), font(30)
    coord = lambda x, y: (left+(x+.25)/.5*(right-left),bottom-y/5*(bottom-top))
    for n in range(6):
        _, y = coord(0,n)
        d.line((left,y,right,y),fill='#d7e1e6',width=2)
        d.text((left-20,y),str(n),font=tick,fill='#304b59',anchor='rm')
    for x in (-.25,-.125,0,.125,.25):
        xx,_=coord(x,0)
        d.line((xx,top,xx,bottom),fill='#d7e1e6',width=2)
        d.text((xx,bottom+20),f'{x:g}',font=tick,fill='#304b59',anchor='mt')
    d.line((left,top,left,bottom,right,bottom),fill='#304b59',width=4)
    for i,(mat,name,color) in enumerate([('wall','墙体','#285b8d'),('metal','金属板','#a05217')]):
        vals = [(float(r['origin_perturbation_ns']),float(r['mae_cm']))
                for r in summary['clock_sensitivity'] if r['material']==mat]
        pts = [coord(x,y) for x,y in vals]
        d.line(pts,fill=color,width=7)
        for x,y in pts:
            if i==0:
                d.ellipse((x-9,y-9,x+9,y+9),fill='white',outline=color,width=5)
            else:
                d.rectangle((x-8,y-8,x+8,y+8),fill='white',outline=color,width=5)
        x = 560 + i*260
        d.line((x,735,x+54,735),fill=color,width=6)
        d.text((x+72,735),name,font=tick,fill='#304b59',anchor='lm')
    d.text((left,28),'平均绝对误差（cm）',font=label,fill='#172e41',anchor='lm')
    d.text(((left+right)/2,667),'公共时间原点扰动（ns）',font=label,fill='#172e41',anchor='mm')
    im.save(destination,dpi=(400,400),optimize=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--results',type=Path,default=HERE/'results')
    cli.add_argument('--calibration',type=Path,default=HERE.parent/'A题-1.1通道校正'/'results')
    cli.add_argument('--output',type=Path,default=HERE/'results')
    cli.add_argument('--font',type=Path,default=SYSTEM_FONT)
    cli.add_argument('--font-index',type=int,default=1)
    opt=cli.parse_args()
    opt.output.mkdir(parents=True,exist_ok=True)
    summary=diagnostics(opt.results,opt.calibration)
    (opt.output/'paper_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    font=lambda size:ImageFont.truetype(str(opt.font),size,index=opt.font_index)
    flow_diagram(opt.output/'paper_workflow.png',font)
    clock_plot(opt.output/'paper_clock_sensitivity.png',summary,font)


if __name__=='__main__':
    main()
