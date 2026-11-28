"""A 题 1.1 接收通道的等效相对校正。Python 3.10+，仅依赖 NumPy。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import warnings

import numpy as np

from radar_reader import (
    BAND_HZ, CHANNELS, N_SAMPLES_PER_SWEEP, SAMPLE_RATE_HZ, TX, Group, discover_groups, inventory_rows, read_group,
)

N = N_SAMPLES_PER_SWEEP
FS = SAMPLE_RATE_HZ
FS_PER_NS = FS / 1e9
C_CM_PER_NS = 29.9792458
FREQUENCY = np.arange(N // 2 + 1) * (FS // N)
BAND = (FREQUENCY >= BAND_HZ[0]) & (FREQUENCY <= BAND_HZ[1])
FD = FREQUENCY[BAND] / 1e9
FR = FS_PER_NS - FD
CORE = (FD >= 0.48) & (FD <= 1.52)
INTERIOR = (FD >= 0.40) & (FD <= 1.60)
RADIAL = np.array([np.hypot(antenna.x_cm - TX.x_cm, antenna.z_cm - TX.z_cm) for antenna in CHANNELS])
TIMES = np.arange(N)
CLIP = 32764
LAGS = np.arange(-2.50, 2.501, 0.01)
PHASORS = np.exp(2j * np.pi * FD[::5, None] * LAGS)
SAMPLE_BINS = slice(None, None, 5)

t = np.arange(N) / FS
REFERENCE = np.fft.rfft(np.hamming(N) * np.cos(2 * np.pi * (0.30e9 * t + 0.5 * 1.4e15 * t**2)))
PULSE_COMP = np.zeros_like(REFERENCE)
PULSE_COMP[BAND] = np.hanning(BAND.sum()) * np.conj(REFERENCE[BAND]) / np.abs(REFERENCE[BAND])
INVERSE = np.zeros_like(REFERENCE)
EDGE_WEIGHT = np.clip(np.minimum(FD - FD[0], FD[-1] - FD) / 0.16, 0, 1)
INVERSE[BAND] = EDGE_WEIGHT / REFERENCE[BAND]


@dataclass
class Observation:
    group: Group
    packets: tuple[int, ...]
    sample_audit: list[dict]
    clipped: np.ndarray
    peaks: np.ndarray
    impulses: np.ndarray
    repeat_error: np.ndarray


def quiet_median(a, axis=0):
    # A channel can be wholly clipped in one group. Preserve its missing values.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def circular_difference(a):
    return (a + N / 2) % N - N / 2


def parabolic_peak(a):
    k = np.argmax(a, axis=-1)
    row = np.arange(a.shape[0])
    lower, center, upper = a[row, (k - 1) % N], a[row, k], a[row, (k + 1) % N]
    curvature = lower + upper - 2 * center
    offset = np.divide(0.5 * (lower - upper), curvature, out=np.zeros_like(center), where=curvature < 0)
    return k + offset


def observe(group):
    audit = {}
    data, packet_ids = read_group(group, audit)
    clipped = (data >= CLIP).sum(axis=-1) + (data <= -CLIP).sum(axis=-1)
    cycles = data.reshape(len(packet_ids), len(CHANNELS), -1, N).astype(np.float64)
    average = cycles.mean(axis=2)
    spread = np.sqrt(np.mean((cycles - average[:, :, None]) ** 2, axis=(2, 3)))
    rms = np.sqrt(np.mean(average**2, axis=-1))
    repeat_error = np.divide(spread, rms, out=np.full_like(spread, np.nan), where=rms > 0)
    spectra = np.fft.rfft(average, axis=-1)
    compressed = np.fft.ifft(spectra * PULSE_COMP, n=N, axis=-1)
    peaks = np.stack([parabolic_peak(np.abs(p)) for p in compressed])
    impulses = np.fft.ifft(spectra * INVERSE, n=N, axis=-1)
    rows = [
        {
            "group_id": group.id, "array_row": i + 1,
            "recorded_packet_ids": "|".join(map(str, audit["row_packet_ids"][i])),
            "shared_packet_ids": "|".join(map(str, packet_ids)),
            "excluded_packet_ids": "|".join(map(str, audit["unmatched_packet_ids"][i])),
            "samples_per_channel_per_packet": data.shape[-1], "matched_channels": data.shape[1],
        }
        for i in range(len(group.files))
    ]
    return Observation(group, packet_ids, rows, clipped, peaks, impulses, repeat_error)


def choose_reference(observations):
    air = [o for o in observations if o.group.material == "air"]
    peaks = np.concatenate([np.max(np.abs(o.impulses), axis=-1) for o in air])
    lower_strength = np.quantile(peaks, 0.10, axis=0)
    clipped = np.sum(np.concatenate([o.clipped > 0 for o in observations]), axis=0)
    candidates = np.flatnonzero(clipped == 0)
    if not candidates.size:
        raise ValueError("校准组中没有未出现限幅的接收通道")
    ref = int(candidates[np.argmax(lower_strength[candidates])])
    order = sorted(range(len(CHANNELS)), key=lambda i: (clipped[i] > 0, -lower_strength[i]))
    rows = [{
        "rx_id": CHANNELS[i].id, "reference_candidate_rank": rank, "calibration_packets_with_clipping": int(clipped[i]),
        "air_echo_strength_p10": number(lower_strength[i]), "selected": int(i == ref),
    } for rank, i in enumerate(order, 1)]
    return ref, rows


def air_prior(observations, ref):
    shifts = []
    delay_geom = (RADIAL - RADIAL[ref]) / C_CM_PER_NS
    for observation in observations:
        good = (observation.clipped == 0) & (observation.clipped[:, ref, None] == 0)
        difference = circular_difference(observation.peaks - observation.peaks[:, ref, None])
        shifts.append(quiet_median(np.where(good, difference / FS_PER_NS - delay_geom, np.nan)))
    prior = quiet_median(np.array(shifts))
    if not np.all(np.isfinite(prior)):
        raise ValueError("空场粗时延无法覆盖全部通道")
    return prior


def reflected_profile(observation, prior, ref):
    # Reflection to a plane at d is equivalent to a direct path from the mirror Tx.
    d = observation.group.distance_cm
    path = np.hypot(RADIAL, 2 * d)
    geometry = (path - path[ref]) / C_CM_PER_NS
    arrival = observation.peaks[:, ref, None] + FS_PER_NS * (
        (path[None, :] - RADIAL[ref]) / C_CM_PER_NS + prior
    )
    relative_samples = circular_difference(TIMES[None, None, :] - arrival[:, :, None])
    gate = np.cos(0.5 * np.pi * np.clip((np.abs(relative_samples) - 3) / 2, 0, 1)) ** 2
    spectrum = np.fft.fft(observation.impulses * gate, axis=-1)[..., :FREQUENCY.size][..., BAND]
    amplitude = np.abs(spectrum)
    # Per-packet strength screen, relative to each gated echo's own passband.
    # This is a spectral support screen, not a measurement of thermal-noise SNR.
    scale = np.quantile(amplitude[:, :, CORE], 0.9, axis=-1)[:, :, None]
    strong = (amplitude > 0.08 * scale) & (amplitude > 0) & (observation.clipped[:, :, None] == 0)
    valid = strong & strong[:, ref, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (spectrum / spectrum[:, ref, None, :]) * (path / path[ref])[None, :, None]
    ratio *= np.exp(-2j * np.pi * geometry[:, None] * FR)
    ratio = np.where(valid, ratio, np.nan + 1j * np.nan)
    magnitude = np.abs(ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gain = quiet_median(np.log(magnitude))
        phase_sum = np.nansum(ratio / magnitude, axis=0)
    usable = np.isfinite(log_gain) & (np.abs(phase_sum) > 0)
    return np.where(usable, np.exp(log_gain + 1j * np.angle(phase_sum)), np.nan + 1j * np.nan)


def describe(profiles):
    profiles = np.array(profiles)
    gain = np.abs(profiles)
    count = np.isfinite(gain).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gain = quiet_median(np.log(gain))
        circular_mean = np.nansum(profiles / gain, axis=0) / count
        dispersion_db = (20 / np.log(10)) * quiet_median(np.abs(np.log(gain) - log_gain))
    return log_gain, circular_mean, count, dispersion_db


def fit_one_delay(response, weights):
    """Search the slope of circular phase with a free intercept, in ns."""
    samples = response[SAMPLE_BINS]
    weights = weights[SAMPLE_BINS]
    good = (weights > 0) & np.isfinite(samples) & (np.abs(samples) > 0)
    f = FD[SAMPLE_BINS][good]
    if f.size < 14 or np.ptp(f) < 0.30:
        return np.nan, np.nan, False
    unit = samples[good] / np.abs(samples[good])
    w = weights[good]
    values = np.abs((w * unit) @ PHASORS[good]) / w.sum()
    idx = int(np.argmax(values))
    if idx in (0, len(LAGS) - 1):
        return float(LAGS[idx]), float(values[idx]), False
    left, center, right = values[idx - 1 : idx + 2]
    curvature = left + right - 2 * center
    fraction = 0.5 * (left - right) / curvature if curvature < 0 else 0
    return float(LAGS[idx] + fraction * 0.01), float(center), True


def fit_reference_model(wall, metal, ref):
    wg, wp, wn, wd = describe(wall)
    mg, mp, mn, md = describe(metal)
    # Each material has equal weight; each acquisition group contributes once.
    log_gain = (wg + mg) / 2
    phase = (wp + mp) / 2
    gain = np.exp(log_gain)
    mean_phase = np.angle(phase)
    disagreement = np.abs(np.angle(wp * mp.conj(), deg=True))
    gain_disagreement = np.abs(wg - mg) * (20 / np.log(10))
    measured = (wn >= 4) & (mn >= 4) & np.isfinite(gain) & np.isfinite(phase)
    supported = (
        measured & INTERIOR & (np.abs(wp) >= 0.55) & (np.abs(mp) >= 0.55)
        & (disagreement <= 70) & (gain_disagreement <= 8) & (wd <= 5) & (md <= 5)
        & (gain >= 1 / 5) & (gain <= 5)
    )
    weights = (CORE & measured).astype(float)
    stable = []
    delay = []
    coherence = []
    material_delay_gap = []
    status = []
    for i in range(len(CHANNELS)):
        estimate, strength, interior = fit_one_delay(phase[i], weights[i])
        screen = weights[i]
        dw, _, wall_interior = fit_one_delay(wp[i], screen)
        dm, _, metal_interior = fit_one_delay(mp[i], screen)
        delta = abs(dw - dm)
        good = interior and strength >= 0.70 and wall_interior and metal_interior and delta <= 0.75
        if not np.isfinite(estimate):
            estimate, strength, _ = fit_one_delay(phase[i], screen)
        delay.append(estimate)
        coherence.append(strength)
        stable.append(good)
        material_delay_gap.append(delta)
        reason = "ok" if good else "delay_or_material_inconsistent"
        status.append(reason if np.isfinite(estimate) else "insufficient_delay_support")
    delay = np.array(delay)
    stable = np.array(stable)
    delay[ref], stable[ref], status[ref] = 0, True, "reference"
    gain[ref], mean_phase[ref] = 1, 0
    # exp(+j2π f_D Δtau) advances a late channel. The residual is the
    # phase remaining after that advance. Its separate correction is weighted.
    residual = np.angle(np.exp(1j * (mean_phase + 2 * np.pi * FD * delay[:, None])))
    # Preserve the broad central band across caution bins. The table still
    # distinguishes these bins from those passing the stronger support test.
    enabled = measured & INTERIOR & stable[:, None] & (gain >= 0.2) & (gain <= 5)
    return {
        "gain": gain, "residual": residual, "delay": delay, "stable": stable,
        "coherence": coherence, "delta": material_delay_gap, "status": status, "wall_count": wn,
        "metal_count": mn, "phase_gap": disagreement,
        "gain_gap": gain_disagreement, "observed": measured, "reliable": supported,
        "enabled": enabled,
    }


def correction_for(model, residual_weight):
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = 2 * np.pi * FD * model["delay"][:, None] - residual_weight * model["residual"]
        inverse = np.exp(1j * phase) / model["gain"]
    return np.where(model["enabled"], inverse, 1 + 0j)


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, digits=7):
    return format(float(value), f".{digits}g") if np.isfinite(value) else ""


def delay_rows(cal, prior, observations, ref):
    for i, antenna in enumerate(CHANNELS):
        applied = np.abs(cal["correction"][i] - 1) > 1e-12
        yield {
            "rx_id": antenna.id, "array_row": antenna.row, "file_suffix": antenna.row - 1,
            "position_in_row": antenna.in_row, "physical_column": antenna.column, "x_from_tx_cm": antenna.x_cm,
            "height_cm": antenna.z_cm, "reference_rx_id": CHANNELS[ref].id,
            "air_coarse_delay_ns": number(prior[i]), "estimated_delay_ns": number(cal["delay"][i]),
            "applied_time_shift_ns_in_enabled_band": number(-cal["delay"][i] if applied.any() else 0),
            "delay_fit_coherence": number(cal["coherence"][i]),
            "wall_metal_delay_difference_ns": number(cal["delta"][i]),
            "reliable_frequency_bins": int(cal["reliable"][i].sum()),
            "enabled_frequency_bins": int(cal["enabled"][i].sum()),
            "calibration_packets_with_clipping": sum(int(np.count_nonzero(obs.clipped[:, i])) for obs in observations),
            "status": cal["status"][i],
        }


def freq_rows(cal, ref):
    for i, antenna in enumerate(CHANNELS):
        for k, f in enumerate(FD):
            state = "applied"
            if antenna.id == CHANNELS[ref].id:
                state = "reference"
            elif not INTERIOR[k]:
                state = "sweep_band_edge"
            elif not cal["observed"][i, k]:
                state = "insufficient_signal_or_groups"
            elif not cal["enabled"][i, k]:
                state = "unstable_delay_or_extreme_gain"
            elif not cal["reliable"][i, k]:
                state = "applied_with_caution"
            c = cal["correction"][i, k]
            yield {
                "rx_id": antenna.id, "reference_rx_id": CHANNELS[ref].id,
                "f_digital_GHz": f"{f:.3f}", "f_physical_GHz": f"{FR[k]:.3f}",
                "amplitude_ratio_rx_over_reference": number(cal["gain"][i, k]),
                "residual_digital_phase_deg": number(np.degrees(cal["residual"][i, k])),
                "applied_magnitude_multiplier": number(abs(c)),
                "applied_residual_phase_deg": number(-cal["residual_weight"] * np.degrees(cal["residual"][i, k])
                                                     if cal["enabled"][i, k] and i != ref else 0),
                "applied_total_digital_phase_deg": number(np.angle(c, deg=True)),
                "applied_complex_real": number(c.real), "applied_complex_imag": number(c.imag),
                "wall_groups_with_signal": int(cal["wall_count"][i, k]),
                "metal_groups_with_signal": int(cal["metal_count"][i, k]),
                "wall_metal_phase_gap_deg": number(cal["phase_gap"][i, k]),
                "wall_metal_gain_gap_db": number(cal["gain_gap"][i, k]),
                "enough_signal_in_both_materials": int(cal["observed"][i, k]),
                "frequency_support_pass": int(cal["reliable"][i, k]),
                "apply_parameter": int(cal["enabled"][i, k] and i != ref), "status": state,
            }


def _errors(profile, mask):
    d, _, reliable = fit_one_delay(profile, mask.astype(float))
    if np.count_nonzero(mask) < 20:
        return (np.nan, np.nan, np.nan)
    mag = np.median(np.abs(20 * np.log10(np.abs(profile[mask]))))
    if reliable:
        flattened = profile[mask] * np.exp(2j * np.pi * FD[mask] * d)
        phase = np.median(np.abs(np.angle(flattened, deg=True)))
        return (abs(d), mag, phase)
    return (np.nan, mag, np.nan)


def validate(observations, profiles, model, ref):
    details = []
    values = {}
    for observation, profile in zip(observations, profiles):
        group = observation.group
        for i, antenna in enumerate(CHANNELS):
            if i == ref:
                continue
            # Same observable central bins before/after, including bins left at
            # multiplier 1. A support-only evaluation would hide missed channels.
            mask = CORE & np.isfinite(profile[i])
            before = _errors(profile[i], mask)
            after = _errors(profile[i] * model["correction"][i], mask)
            values[group.id, i] = (before, after)
            details.append({
                "group_id": group.id, "material": group.material, "distance_cm": group.distance_cm,
                "rx_id": antenna.id, "common_eval_frequency_bins": int(mask.sum()),
                "corrected_eval_frequency_bins": int(np.count_nonzero(mask & model["enabled"][i])),
                "clipped_packets": int(np.count_nonzero(observation.clipped[:, i])),
                **metric_fields(np.array(before), np.array(after)),
            })
    return details, values


METRICS = ("abs_delay_ns", "abs_magnitude_db", "abs_residual_phase_deg")


def metric_fields(before, after):
    return {
        f"{key}_{name}": number(value)
        for key, errors in (("before", before), ("after", after))
        for name, value in zip(METRICS, errors)
    }


def paired_medians(pairs):
    a = np.array(pairs, dtype=float)
    common = np.all(np.isfinite(a), axis=1)
    use = np.where(common[:, None, :], a, np.nan)
    before, after = quiet_median(use)
    return before, after, common.sum(axis=0)


def summary_rows(observations, values, model, ref):
    per_channel = []
    aggregate = []
    limits = np.array([0.02, 0.10, 2.0])
    for material in ("wall", "metal"):
        groups = [o.group for o in observations if o.group.material == material]
        a = {}
        for i, antenna in enumerate(CHANNELS):
            if i == ref:
                continue
            pairs = [values[group.id, i] for group in groups]
            before, after, n = paired_medians(pairs)
            a[i] = pairs
            states = np.where(n == 0, "unavailable", np.where(
                before - after >= limits, "improved", np.where(after - before >= limits, "worse", "little_change")
            ))
            per_channel.append({
                "material": material, "rx_id": antenna.id, "model_status": model["status"][i],
                **metric_fields(before, after),
                **{f"paired_groups_{name}": int(value) for name, value in zip(METRICS, n)},
                **{f"change_{name}": value for name, value in zip(METRICS, states)},
            })
        # Median across channels within each group, then across groups so all
        # held-out acquisitions are weighted equally.
        group_metrics = [paired_medians([values[group.id, i] for i in a]) for group in groups]
        before, after, paired = paired_medians([(b, c) for b, c, _ in group_metrics])
        status_rows = [row for row in per_channel if row["material"] == material]
        for k, name in enumerate(METRICS):
            states = [row[f"change_{name}"] for row in status_rows]
            aggregate.append({
                "material": material, "metric": name, "paired_groups": int(paired[k]),
                "median_before": number(before[k]), "median_after": number(after[k]),
                "channels_improved": states.count("improved"), "channels_little_change": states.count("little_change"),
                "channels_worse": states.count("worse"), "channels_unavailable": states.count("unavailable"),
                "little_change_below": limits[k],
            })
    return per_channel, aggregate


def select_residual_weight(observations, wall, metal, ref):
    """Compare no, half and full residual rotation by calibration-only folds."""
    group_info = [o.group for o in observations if o.group.material != "air"]
    material = np.array([g.material for g in group_info])
    distance = np.array([g.distance_cm for g in group_info])
    profiles = np.concatenate([wall, metal])
    candidates = (0.0, 0.5, 1.0)
    before_scores = {"wall": [], "metal": []}
    after_scores = {w: {"wall": [], "metal": []} for w in candidates}
    for k, (m, d) in enumerate(zip(material, distance)):
        included = (material != m) | (distance != d)
        fold = fit_reference_model(profiles[included & (material == "wall")],
                                   profiles[included & (material == "metal")], ref)
        p = profiles[k]
        indices = [i for i in range(len(CHANNELS)) if i != ref]
        masks = [CORE & np.isfinite(p[i]) for i in indices]
        fold_before = np.array([_errors(p[i], mask) for i, mask in zip(indices, masks)])
        fold_after = []
        for weight in candidates:
            c = correction_for(fold, weight)
            fold_after.append(np.array([_errors(p[i] * c[i], mask) for i, mask in zip(indices, masks)]))
        common = np.isfinite(fold_before) & np.all(np.isfinite(fold_after), axis=0)
        before_scores[m].append(quiet_median(np.where(common, fold_before, np.nan)))
        for weight, errors in zip(candidates, fold_after):
            after_scores[weight][m].append(quiet_median(np.where(common, errors, np.nan)))
    rows, scores = [], {}
    for weight in candidates:
        ratios = []
        for m in ("wall", "metal"):
            before, after, paired = paired_medians(list(zip(before_scores[m], after_scores[weight][m])))
            ratios.extend(after / before)
            for j, metric in enumerate(METRICS):
                rows.append({
                    "residual_weight": weight, "material": m, "metric": metric, "paired_groups": int(paired[j]),
                    "median_before": number(before[j]), "median_after": number(after[j]),
                    "after_over_before": number(after[j] / before[j]),
                })
        scores[weight] = float(np.mean(ratios))
    selected = min(scores, key=scores.get)
    for row in rows:
        weight = row["residual_weight"]
        row.update(mean_normalized_error=number(scores[weight]), selected=int(weight == selected))
    return selected, rows


def verification(model, ref):
    if not np.allclose(model["correction"][ref], 1, atol=1e-12):
        raise ValueError("参照通道应当是单位校正")
    if not np.all(np.isfinite(model["correction"])):
        raise ValueError("校正因子出现非有限数")
    # Delay sign and conjugation check, at the same frequencies as the table.
    tau, geometry, amplitude, offset = 0.37, 0.28, 1.6, 0.55
    physical = amplitude * np.exp(2j * np.pi * FR * geometry) * np.exp(1j * (offset - 2 * np.pi * FD * tau))
    hardware = physical * np.exp(-2j * np.pi * FR * geometry)
    recovered, _, good = fit_one_delay(hardware, CORE.astype(float))
    corrected = hardware * np.exp(2j * np.pi * FD * recovered - 1j * offset) / amplitude
    if not good or abs(recovered - tau) > 1e-5 or not np.allclose(corrected, 1, atol=1e-4):
        raise ValueError("合成信号的时延符号或射频相位换算未通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    groups = discover_groups(args.project)
    args.output.mkdir(exist_ok=True, parents=True)

    split = [{
        "group_id": g.id, "material": g.material, "distance_cm": g.distance_cm, "timestamp": g.timestamp,
        "split": g.split, "role": "air_timing_prior" if g.material == "air" else
        ("reflection_fit" if g.split == "calibration" else "evaluation_only"),
    } for g in groups]
    write_rows(args.output / "group_split.csv", split[0], split)
    inputs = inventory_rows(groups, args.project)
    write_rows(args.output / "input_files.csv", inputs[0], inputs)

    # Read the held-out waveforms only after the model and parameters are written.
    train = [observe(g) for g in groups if g.split == "calibration"]
    air = [obs for obs in train if obs.group.material == "air"]
    ref, reference_screen = choose_reference(train)
    write_rows(args.output / "reference_screen.csv", reference_screen[0], reference_screen)
    prior = air_prior(air, ref)
    wall = [reflected_profile(obs, prior, ref) for obs in train if obs.group.material == "wall"]
    metal = [reflected_profile(obs, prior, ref) for obs in train if obs.group.material == "metal"]
    residual_weight, training_cv = select_residual_weight(train, wall, metal, ref)
    write_rows(args.output / "calibration_crosscheck.csv", training_cv[0], training_cv)
    model = fit_reference_model(wall, metal, ref)
    model["residual_weight"] = residual_weight
    model["correction"] = correction_for(model, residual_weight)
    verification(model, ref)
    delays = list(delay_rows(model, prior, train, ref))
    write_rows(args.output / "channel_delays.csv", delays[0], delays)
    frequency_rows = freq_rows(model, ref)
    first = next(frequency_rows)
    write_rows(args.output / "frequency_correction.csv", first, chain([first], frequency_rows))

    held = [observe(g) for g in groups if g.split == "holdout"]
    profiles = [reflected_profile(obs, prior, ref) for obs in held]
    details, values = validate(held, profiles, model, ref)
    write_rows(args.output / "holdout_by_group_channel.csv", details[0], details)
    channels, summary = summary_rows(held, values, model, ref)
    write_rows(args.output / "holdout_by_channel.csv", channels[0], channels)
    write_rows(args.output / "holdout_summary.csv", summary[0], summary)
    packets = [entry for obs in train + held for entry in obs.sample_audit]
    write_rows(args.output / "packet_audit.csv", packets[0], packets)
    signals = [{
        "group_id": obs.group.id, "material": obs.group.material, "split": obs.group.split,
        "matched_packets": len(obs.packets), "clipped_channel_packets": int(np.count_nonzero(obs.clipped)),
        "reference_clipped_packets": int(np.count_nonzero(obs.clipped[:, ref])),
        "reference_direct_peak_median_sample": number(np.median(obs.peaks[:, ref])),
        "median_period_rms_difference_over_signal": number(np.median(obs.repeat_error)),
        "p90_period_rms_difference_over_signal": number(np.quantile(obs.repeat_error, 0.9)),
    } for obs in train + held]
    write_rows(args.output / "signal_audit.csv", signals[0], signals)
    print(f"{len(groups)} groups; {len(inputs)} files; {sum(len(o.packets) for o in train + held)} matched packets")
    print(f"{len(delays)} channels; {len(FD)} frequency bins/channel; {model['stable'].sum() - 1}/62 stable non-reference delays")
    print(f"Reference channel: {CHANNELS[ref].id}")
    print(f"Residual phase weight from calibration-only crosscheck: {residual_weight}")
    print(f"Applied gains/phases: {np.count_nonzero(model['enabled'][np.arange(len(CHANNELS)) != ref])} non-reference bins")
    for row in summary:
        print(row["material"], row["metric"], row["median_before"], "->", row["median_after"],
              "improved/little change/worse/missing:",
              [row[key] for key in ("channels_improved", "channels_little_change", "channels_worse", "channels_unavailable")])


if __name__ == "__main__":
    main()
