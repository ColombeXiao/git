"""Process the local Problem A fixtures and reproduce all numeric outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'A题-1.1通道校正'))
import calibrate_q11 as q11
from radar_reader import CHANNELS, FILE_PATTERN, Group, TX, discover_groups, inventory_rows, read_group

N = q11.N
C = q11.C_CM_PER_NS
T = np.arange(-32, 139)
TINDEX = T % N
F = q11.FREQUENCY / 1e9
GEOM = np.array([[b.x_cm, 0.0, b.z_cm] for b in CHANNELS])
TX0 = np.array([TX.x_cm, 0.0, TX.z_cm])
RHO = np.linalg.norm(GEOM - TX0, axis=1)
IDS = {b.id: i for i, b in enumerate(CHANNELS)}
ANCHORS = np.array([IDS[x] for x in ('R05C07', 'R03C09', 'R04C06', 'R04C10')])
DISTANCES = np.arange(20.0, 155.01, .25)
# 5-cm scan increments, positive toward the first physical column.
WALL_X = np.arange(-20, 141, 5)
METAL_X = np.arange(0, 131, 5)
GAIN_REFERENCE_CM = 80.0
GAIN_EXPONENT = 1.2
GAIN_CAP = 3.0


@dataclass
class Frame:
    group: Group
    packets: tuple[int, ...]
    valid: np.ndarray
    raw: np.ndarray
    corrected: np.ndarray
    direct: np.ndarray
    full: np.ndarray
    timing: np.ndarray
    omitted_row_packets: int


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_out(path, rows):
    q11.write_rows(path, rows[0], rows)


def load_q11(path):
    rows = read_csv(path / 'channel_delays.csv')
    ref = IDS[next(x['rx_id'] for x in rows if x['status'] == 'reference')]
    stable = np.array([row['status'] in ('ok', 'reference') for row in rows])
    correction = np.ones((len(CHANNELS), len(F)), dtype=complex)
    seen = set()
    for row in read_csv(path / 'frequency_correction.csv'):
        i = IDS[row['rx_id']]
        k = round(float(row['f_digital_GHz']) * 1000)
        seen.add((i, k))
        correction[i, k] = complex(float(row['applied_complex_real']), float(row['applied_complex_imag']))
    expected = {(i, k) for i in IDS.values() for k in range(300, 1701)}
    if seen != expected or not np.all(np.isfinite(correction)):
        raise ValueError('Incomplete Q1.1 frequency table')
    return ref, stable, correction


def attachment_path():
    dirs = sorted(ROOT.glob('A 题*/附件'))
    if len(dirs) != 1:
        raise ValueError(f'Expected one raw attachment directory, found {dirs}')
    return dirs[0]


def imaging_groups(scene):
    title = {'walls': '墙体', 'metal_objects': '金属柱和金属铲'}[scene]
    parent = attachment_path() / '附件3-成像数据' / title
    timestamps = defaultdict(dict)
    for path in sorted(parent.glob('*.dat')):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f'Malformed scan file name: {path}')
        timestamps[match['time']][int(match['row'])] = path
    gaps = []
    groups = []
    for time, rows in sorted(timestamps.items()):
        if set(rows) != set(range(8)) or any(p.stat().st_size == 0 for p in rows.values()):
            gaps += [{'scene': scene, 'timestamp': time, 'reason': 'incomplete_or_empty_scan',
                      'relative_path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size} for p in rows.values()]
            continue
        groups.append(Group(scene, time, None, 'imaging', time, tuple(rows[k] for k in range(8))))
    expected = WALL_X if scene == 'walls' else METAL_X
    if len(groups) != len(expected):
        raise ValueError(f'{scene}: got {len(groups)} complete positions, expected {len(expected)}')
    return groups, gaps


def rms(a, axis=-1):
    return np.sqrt(np.mean(np.abs(a)**2, axis=axis))


def med(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def cmed(a, axis=0):
    return med(a.real, axis) + 1j * med(a.imag, axis)


def rfft_packets(group):
    audit = {}
    data, ids = read_group(group, audit)
    usable = np.max(np.abs(data.astype(np.int32)), axis=-1) < q11.CLIP
    r = data.reshape(len(ids), len(CHANNELS), -1, N).mean(axis=2)
    excluded = sum(map(len, audit['unmatched_packet_ids']))
    return ids, np.fft.rfft(r, axis=-1), usable, excluded


def shift_spectrum(s, advance_samples):
    return s * np.exp(2j * np.pi * np.asarray(advance_samples)[..., None] * F / 4)


def pulse(s):
    return np.fft.ifft(s * q11.INVERSE, n=N, axis=-1)


def fit_anchor_template(z):
    p = pulse(z[ANCHORS])
    center = q11.parabolic_peak(np.abs(p))
    circular = q11.circular_difference(np.arange(N)[None] - center[:, None])
    w = np.cos(np.pi / 2 * np.clip((np.abs(circular) - 4) / 4, 0, 1))**2
    return np.conj(np.fft.fft(p*w, axis=-1)[..., :len(F)])


def common_clock(z, usable, kernel, template_zero):
    correlation = pulse(z[:, ANCHORS]*kernel[None])
    scale = np.max(np.abs(correlation), axis=-1)
    scores = np.divide(np.abs(correlation), scale[..., None], out=np.zeros_like(correlation.real),
                       where=scale[..., None] > 0)
    good = usable[:, ANCHORS]
    if np.any(good.sum(axis=1) < 2):
        raise ValueError('Fewer than 2 timing anchors in a matched packet')
    score = np.sum(scores * good[..., None], axis=1)
    return q11.circular_difference(q11.parabolic_peak(score) - template_zero)


def air_frame(air, ref, correction):
    packets = [rfft_packets(g) for g in air]
    _, z, _, _ = packets[0]
    origin = q11.parabolic_peak(np.abs(pulse(z[0, ref:ref+1])))[0]
    seed = shift_spectrum(z[0], origin)
    kernel = fit_anchor_template(seed)
    zero = common_clock(seed[None], np.ones((1,len(GEOM)),bool), kernel, 0)[0]
    windows = []
    for _, spectra, good, _ in packets:
        offset = common_clock(spectra, good, kernel, zero)
        shifted = shift_spectrum(spectra, offset[:, None])
        windows.append(cmed(np.where(good[..., None], shifted, np.nan+1j*np.nan)))
    spectrum = cmed(np.stack(windows))
    raw = pulse(spectrum)[..., TINDEX]
    corrected = pulse(spectrum*correction)[..., TINDEX]
    centers = T[np.argmax(np.where((T >= -8) & (T <= 13), np.abs(corrected), 0), axis=-1)]
    # A short, asymmetric direct-path fit precedes the expected target return.
    delta = T[None] - centers[:, None]
    fit = np.cos(np.pi/2 * np.clip((np.abs(delta+1) - 2.5) / 2.0,0,1))**2
    direct = np.cos(np.pi/2 * np.clip((delta - 10) / 9, 0, 1))**2
    return {'kernel':kernel,'zero':zero,'raw':raw,'corrected':corrected,'centers':centers,'fit':fit,'direct':direct}


def process(group, template, correction):
    ids, z, usable, excluded = rfft_packets(group)
    offsets = common_clock(z, usable, template['kernel'], template['zero'])
    shifted = shift_spectrum(z, offsets[:, None])
    raw = pulse(shifted)[..., TINDEX]
    calibrated = pulse(shifted * correction)[..., TINDEX]
    air = template['corrected']
    gate = template['fit']
    denom = np.sum(gate*np.abs(air)**2,axis=1)
    gain = np.divide(np.sum(calibrated * np.conj(air) * gate,axis=-1), denom,
                     out=np.zeros((len(ids),len(GEOM)),complex),where=denom>0)
    # The coefficient is estimated only from the direct-wave window.
    background = gain[...,None]*air
    residual_direct = calibrated - background*template['direct']
    residual_full = calibrated - background
    # Rotate each packet by its common direct-wave phase before averaging.
    phase = np.angle(np.sum(np.where(usable[:,ANCHORS],gain[:,ANCHORS],0),axis=-1))
    ramp = np.exp(-1j*phase[:,None,None])
    def average(trace):
        value = cmed(np.where(usable[...,None], trace*ramp, np.nan+1j*np.nan))
        return np.where(np.isfinite(value),value,0)
    return Frame(group,ids,usable.any(axis=0),average(raw),average(calibrated),average(residual_direct),
                 average(residual_full),offsets,excluded)


def interpolate(y, sample):
    """Linear interpolation of a trace at sample offsets relative to T[0]."""
    coordinate = np.clip(sample - T[0], 0, len(T)-1.000001)
    base = coordinate.astype(np.intp)
    low = np.take_along_axis(y, base, axis=-1)
    high = np.take_along_axis(y, base+1, axis=-1)
    return low + (high-low)*(coordinate-base)


def path_samples(d, ref, bias=0.0):
    return 4 * (np.hypot(RHO[:,None],2*np.asarray(d)[None]) - RHO[ref]) / C + bias


def range_profile(frame, trace, use, ref, bias=0.0):
    envelope = np.abs(getattr(frame,trace))
    values = interpolate(envelope,path_samples(DISTANCES,ref,bias))
    # Prevent one strong or near-saturated receiver from owning the peak.
    scale = np.quantile(values[use],.90,axis=-1)
    scale = np.maximum(scale,np.median(scale)*.25)
    normalized = values[use]/scale[:,None]
    return .75*np.median(normalized,axis=0) + .25*np.mean(normalized,axis=0)


def max_position(a,x):
    k = int(np.argmax(a))
    if k == 0 or k == len(x)-1:
        return float(x[k])
    l,m,r = a[k-1:k+2]
    step = x[k+1]-x[k]
    offset = .5*(l-r)/(l-2*m+r) if l-2*m+r < 0 else 0
    return float(x[k]+offset*step)


def infer_bias(frames,trace,stable,ref):
    offsets = []
    for frame in frames:
        d = frame.group.distance_cm
        a = range_profile(frame,trace,frame.valid & stable,ref)
        near = np.abs(DISTANCES - d) <= 14
        peak = max_position(a[near],DISTANCES[near])
        offsets.append(8*(peak-d)/C)
    return float(np.median(offsets))


def select_range_method(training,stable,ref):
    # Separate fit for the common time origin from scoring a withheld distance
    # interval during the calibration-only cross-check.
    traces = ('raw','corrected','direct','full')
    output = []
    for trace in traces:
        errors = defaultdict(list)
        for frame in training:
            other = [x for x in training if (x.group.material,x.group.distance_cm) !=
                     (frame.group.material,frame.group.distance_cm)]
            bias = infer_bias(other,trace,stable,ref)
            p = range_profile(frame,trace,frame.valid & stable,ref,bias)
            estimate = max_position(p,DISTANCES)
            errors[frame.group.material].append(abs(estimate-frame.group.distance_cm))
        for material in ('wall','metal'):
            output.append({'pipeline':trace,'material':material,'groups':len(errors[material]),
                           'mae_cm':float(np.mean(errors[material])),
                           'median_abs_error_cm':float(np.median(errors[material]))})
    allowed = ('direct','full')
    selected = min(allowed,key=lambda t:sum(x['mae_cm'] for x in output if x['pipeline']==t))
    for row in output:
        row['selected'] = int(row['pipeline']==selected)
    return selected,output


def suppress_metrics(frame,template,stable,ref):
    use = frame.valid & stable
    centers = template['centers']
    near_direct = np.abs(T[None]-centers[:,None]) <= 3
    arrival = path_samples(np.array([frame.group.distance_cm]),ref)
    near_target = np.abs(T[None]-arrival) <= 3
    rows = []
    tiny = np.finfo(float).tiny
    for mode in ('direct','full'):
        before, after = np.abs(frame.corrected)**2,np.abs(getattr(frame,mode))**2
        direct = 10*np.log10(np.maximum(np.sum(before*near_direct,axis=1),tiny)/
                            np.maximum(np.sum(after*near_direct,axis=1),tiny))
        target = 10*np.log10(np.maximum(np.sum(after*near_target,axis=1),tiny)/
                            np.maximum(np.sum(before*near_target,axis=1),tiny))
        rows.append({'group_id':frame.group.id,'material':frame.group.material,'distance_cm':frame.group.distance_cm,
                     'pipeline':mode,'channels':int(use.sum()),'direct_reduction_db':float(np.median(direct[use])),
                     'target_window_change_db':float(np.median(target[use]))})
    return rows


def q1_example(output, held, template, stable, ref, correction, validation):
    # Include one improved and one worse wall case, selected by auditing the
    # earlier fixed holdout metrics. They are illustrations of heterogeneity.
    candidates = read_csv(validation/'holdout_by_group_channel.csv')
    good = [row for row in candidates if row['material']=='wall' and row['before_abs_residual_phase_deg']
            and row['after_abs_residual_phase_deg'] and stable[IDS[row['rx_id']]]]
    good.sort(key=lambda x:float(x['after_abs_residual_phase_deg'])-float(x['before_abs_residual_phase_deg']))
    low, high = good[0], next(x for x in reversed(good) if x['rx_id']!=good[0]['rx_id'])
    items = []
    for row in (low,high):
        frame = next(f for f in held if f.group.id == row['group_id'])
        group = frame.group
        n = IDS[row['rx_id']]
        # Match the fixed narrow target gate used for the calibrated study.
        obs = q11.observe(group)
        prior = template['q11_prior']
        path = np.hypot(RHO,2*group.distance_cm)
        arrival = obs.peaks[:,ref,None] + 4*((path[None]-RHO[ref])/C+prior)
        k = next(k for k in range(len(obs.packets)) if not obs.clipped[k,n] and not obs.clipped[k,ref])
        delta = q11.circular_difference(np.arange(N)[None]-arrival[k,[ref,n],None])
        gate = np.cos(np.pi/2*np.clip((np.abs(delta)-3)/2,0,1))**2
        h = np.fft.fft(obs.impulses[k,[ref,n]]*gate,axis=-1)[..., :len(F)]
        geo_delay = (path[n]-path[ref])/C
        h[1] *= np.exp(-2j*np.pi*(4-F)*geo_delay)*(path[n]/path[ref])
        after = h[1]*correction[n]
        time = np.arange(-14,15)
        idx = (int(round(arrival[k,ref]))+time)%N
        def wav(v):
            b = v.copy()
            b[~q11.BAND]=0
            return np.fft.ifft(b,n=N)[idx].real
        fmask = q11.CORE
        items.append({'group_id':group.id,'channel':row['rx_id'],'packet':obs.packets[k],
                      'before_phase_deg':float(row['before_abs_residual_phase_deg']),
                      'after_phase_deg':float(row['after_abs_residual_phase_deg']),
                      'time':time/4,'ref_wave':wav(h[0]),'raw_wave':wav(h[1]),'cal_wave':wav(after),
                      'frequency':q11.FD[fmask],'ref_spec':np.abs(h[0,q11.BAND][fmask]),
                      'pre_spec':np.abs(h[1,q11.BAND][fmask]),'post_spec':np.abs(after[q11.BAND][fmask])})
    out = []
    for item in items:
        for type_,keys in [('waveform', ('time','ref_wave','raw_wave','cal_wave')),
                           ('spectrum', ('frequency','ref_spec','pre_spec','post_spec'))]:
            for x,r,a,b in zip(*(item[k] for k in keys)):
                out.append({'group_id':item['group_id'],'channel':item['channel'],'packet':item['packet'],
                            'kind':type_,'x':x,'reference':r,'before':a,'after':b})
    csv_out(output/'calibration_examples.csv',out)
    return items


def gain_curve(y, gamma=GAIN_EXPONENT, cap=GAIN_CAP):
    return np.minimum(cap,np.maximum(1,np.asarray(y)/GAIN_REFERENCE_CM)**gamma)


def backproject(frames, scan_x, x, y, z, trace, stable, ref, bias, row=None):
    """Bistatic envelope image and a phase-weighted comparison."""
    X,Z,Y = np.meshgrid(x,z,y,indexing='xy')
    px,py,pz = X.ravel(),Y.ravel(),Z.ravel()
    size = len(px)
    total = np.zeros(size)
    total_coh = np.zeros(size)
    weight = np.zeros(size)
    spacing = 4/C
    for frame, sx in zip(frames,scan_x):
        use = frame.valid & stable
        if row is not None:
            use &= np.array([a.row==row for a in CHANNELS])
        if not use.any():
            raise ValueError(f'No receivers at {frame.group.id}')
        receivers = GEOM[use].astype(np.float32)
        data = getattr(frame,trace)[use].astype(np.complex64)
        for a in range(0,size,6000):
            b = min(a+6000,size)
            qx,qy,qz = px[a:b],py[a:b],pz[a:b]
            dt = np.sqrt((qx-sx)**2+qy**2+(qz-TX.z_cm)**2)
            dr = np.sqrt((qx[None]-sx-receivers[:,0,None])**2+qy[None]**2+
                         (qz[None]-receivers[:,2,None])**2)
            distance = dt[None]+dr
            time = (distance-RHO[ref])*spacing+bias
            weights = (qy[None]/np.maximum(dt[None],1))**2*(qy[None]/np.maximum(dr,1))**2
            sample = interpolate(data,time)
            mag = np.abs(sample)
            total[a:b] += np.sum(weights*mag,axis=0)
            # Undo RF folding phase before summing within one aperture position.
            # Between-position magnitude stacking tolerates trigger-phase drift.
            coherent = np.sum(weights*sample*np.exp(-2j*np.pi*time),axis=0)
            total_coh[a:b] += np.abs(coherent)
            weight[a:b] += np.sum(weights,axis=0)
    shape = X.shape
    env = np.divide(total,weight,out=np.zeros_like(total),where=weight>1e-7).reshape(shape)
    coherent = np.divide(total_coh,weight,out=np.zeros_like(total_coh),where=weight>1e-7).reshape(shape)
    return env, .45*env+.55*coherent



def range_sensitivity(frames,mode,stable,ref,bias):
    rows = []
    for perturbation in (-1.0,-0.5,0.0,0.5,1.0):
        for material in ('wall','metal'):
            errors = []
            for frame in frames:
                if frame.group.material != material:
                    continue
                profile = range_profile(frame,mode,frame.valid & stable,ref,bias+perturbation)
                errors.append(abs(max_position(profile,DISTANCES)-frame.group.distance_cm))
            rows.append({'material':material,'origin_perturbation_ns':perturbation/4,'groups':len(errors),
                         'mae_cm':np.mean(errors)})
    return rows


def summarize_range(rows):
    out = []
    for target in ('wall','metal','combined'):
        for mode in ('raw','corrected','direct','full','selected'):
            group = [r for r in rows if (target=='combined' or r['material']==target) and r['split']=='holdout']
            modes = [r['selected_pipeline'] if mode == 'selected' else mode for r in group]
            errors = np.array([r[f'{m}_estimate_cm']-r['distance_cm'] for r,m in zip(group,modes)])
            out.append({'material':target,'pipeline':mode,'n':len(group),'mae_cm':np.mean(abs(errors)),
                        'rmse_cm':np.sqrt(np.mean(errors**2)), 'bias_cm':np.mean(errors),
                        'maximum_abs_error_cm':np.max(abs(errors))})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE/'results')
    p.add_argument('--q11', type=Path, default=ROOT/'A题-1.1通道校正'/'results')
    p.add_argument('--quick', action='store_true', help='Only process Q1 and the horizontal wall section.')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    q1groups = discover_groups(ROOT)
    recorded = read_csv(args.q11/'group_split.csv')
    if [(g.id,g.split) for g in q1groups] != [(r['group_id'],r['split']) for r in recorded]:
        raise ValueError('The Q1.1 training and holdout split differs from the reader inventory')
    ref, stable, correction = load_q11(args.q11)
    air = [g for g in q1groups if g.material=='air']
    template = air_frame(air,ref,correction)
    template['q11_prior'] = q11.air_prior([q11.observe(g) for g in air],ref)
    training = [process(g,template,correction) for g in q1groups if g.split=='calibration' and g.material!='air']
    selected, cv = select_range_method(training,stable,ref)
    csv_out(out/'range_training_cv.csv',cv)
    modes = ('raw','corrected','direct','full')
    biases = {m:infer_bias(training,m,stable,ref) for m in modes}
    held = [process(g,template,correction) for g in q1groups if g.split=='holdout']
    audit, profiles, suppression = [],[],[]
    for frame in training+held:
        g = frame.group
        r = {'group_id':g.id,'material':g.material,'split':g.split,'distance_cm':g.distance_cm,
             'selected_pipeline':selected, 'channels':int(np.sum(frame.valid & stable))}
        for mode in modes:
            curve = range_profile(frame,mode,frame.valid & stable,ref,biases[mode])
            r[f'{mode}_estimate_cm'] = max_position(curve,DISTANCES)
            if mode == selected and g.split=='holdout':
                profiles.append((g.id,curve))
        r['selected_error_cm'] = r[f'{selected}_estimate_cm']-g.distance_cm
        audit.append(r)
        if g.split=='holdout':
            suppression += suppress_metrics(frame,template,stable,ref)
    csv_out(out/'ranges_by_group.csv',audit)
    csv_out(out/'range_holdout_metrics.csv',summarize_range(audit))
    csv_out(out/'range_origin_sensitivity.csv',range_sensitivity(held,selected,stable,ref,biases[selected]))
    csv_out(out/'suppression_by_holdout_group.csv',suppression)
    csv_out(out/'range_profiles_holdout.csv',[{'group_id':g,'candidate_distance_cm':x,'score':s}
                                           for g,c in profiles for x,s in zip(DISTANCES,c)])
    q1_example(out,held,template,stable,ref,correction,args.q11)
    d_example = next(x for x in held if x.group.material=='metal' and x.group.distance_cm==76)
    channel = IDS['R05C07']
    csv_out(out/'suppression_example.csv',[{'time_ns':t/4,'before':abs(a),'after':abs(b),
                                         'air':abs(c)} for t,a,b,c in zip(T,d_example.corrected[channel],
                                         getattr(d_example,selected)[channel],template['corrected'][channel])])
    wall_groups, wall_gaps = imaging_groups('walls')
    metal_groups, metal_gaps = imaging_groups('metal_objects')
    csv_out(out/'imaging_skipped_files.csv',wall_gaps+metal_gaps)
    scan_rows = []
    for groups,xx in ((wall_groups,WALL_X),(metal_groups,METAL_X)):
        for group,x in zip(groups,xx):
            scan_rows.append({'group_id':group.id,'timestamp':group.timestamp,'x_tx_cm':x,
                              'relative_first_row_file':str(group.files[0].relative_to(ROOT))})
    csv_out(out/'scan_positions.csv',scan_rows)
    print('Q1 range pipeline selected from training:', selected, 'bias samples:',biases,flush=True)
    for r in summarize_range(audit):
        if r['pipeline']=='selected':
            print(r, flush=True)

    wf = [process(g,template,correction) for g in wall_groups]
    mf = [] if args.quick else [process(g,template,correction) for g in metal_groups]
    frame_audit = [{'group_id':f.group.id,'matched_packets':len(f.packets),
                    'channels_unclipped_in_at_least_one_packet':int(f.valid.sum()),
                    'excluded_row_packets':f.omitted_row_packets,
                    'median_trigger_offset_samples':float(np.median(f.timing))}
                   for f in wf+mf]
    csv_out(out/'imaging_packet_audit.csv',frame_audit)
    x2, y2 = np.arange(-20,141,2),np.arange(0,251,2)
    env, combined = backproject(wf,WALL_X,x2,y2,np.array([TX.z_cm]),selected,stable,ref,biases[selected],row=4)
    np.savez_compressed(out/'wall_2d.npz',x=x2,y=y2,z=TX.z_cm,envelope=env[0],hybrid=combined[0],
                        enhanced=combined[0]*gain_curve(y2), gain=gain_curve(y2))
    if not args.quick:
        wx, wy, wz = np.linspace(-20,140,55),np.linspace(0,250,85),np.linspace(20,80,21)
        vx, vy, vz = np.linspace(0,130,45),wy,wz
        for name, frames, scans, x, y, z in (('walls',wf,WALL_X,wx,wy,wz),
                                             ('metal',mf,METAL_X,vx,vy,vz)):
            e,h = backproject(frames,scans,x,y,z,selected,stable,ref,biases[selected])
            split_images = []
            for parity in (0,1):
                _, sub = backproject(frames[parity::2],scans[parity::2],x,y,z,selected,stable,ref,biases[selected])
                split_images.append(sub)
            np.savez_compressed(out/f'{name}_3d.npz',x=x,y=y,z=z,envelope=e,hybrid=h,
                                enhanced=h*gain_curve(y),gain=gain_curve(y),
                                even_positions=split_images[0],odd_positions=split_images[1])
            print('Imaged',name,h.shape,'maximum',float(np.max(h)),flush=True)
    metadata = {'reference':CHANNELS[ref].id,'stable_channels':int(stable.sum()),
                'range_pipeline':selected,'common_offsets_samples':biases,
                'scan_counts':{'walls':len(wf),'metal':len(mf)},
                'matched_scan_packets':{'walls':sum(len(f.packets) for f in wf),
                                        'metal':sum(len(f.packets) for f in mf)},
                'gain_exponent':GAIN_EXPONENT,'gain_cap':GAIN_CAP,'gain_reference_cm':GAIN_REFERENCE_CM}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    main()
