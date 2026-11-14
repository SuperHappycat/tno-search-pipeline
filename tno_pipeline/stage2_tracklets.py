"""Stage 2 - intra-night tracklet linking of orphan detections.

Usage
-----
    python -m tno_pipeline.stage2_tracklets --ra 2.0 --dec 0.0
    python -m tno_pipeline.stage2_tracklets --ra 2.0 --dec 0.0 --min-det 2 --workers 4

Reads  <outdir>/<tag>/orphans.parquet
Writes <outdir>/<tag>/tracklets.parquet, one row per tracklet:
    tracklet_id, night, n_det, arc_hr, mjd_ref, ra_ref, dec_ref,
    dra_dt, ddec_dt   [arcsec/hr; dra_dt includes cos(dec), i.e. on-sky east rate]
    rate [arcsec/hr], pa_deg [east of north], ecl_angle_deg, rms [arcsec],
    mag_mean, filters, det_ids (list of Stage 1 ids)
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import Config, ecliptic_position_angle, inverse_tangent_plane

log = logging.getLogger("stage2")


def _candidate_pairs(xy, t, expo, mag, filt, cfg: Config):
    """All detection pairs (i earlier than j) whose separation and timing imply a TNO-like rate."""
    span = t.max() - t.min()
    if span < cfg.min_dt_hr:
        return np.empty((0, 2), dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(cfg.max_rate * span, output_type="ndarray")
    if len(pairs) == 0:
        return pairs
    i, j = pairs[:, 0], pairs[:, 1]
    swap = t[j] < t[i]
    i, j = np.where(swap, j, i), np.where(swap, i, j)
    dt = t[j] - t[i]
    sep = np.hypot(*(xy[j] - xy[i]).T)
    rate = np.divide(sep, dt, out=np.full_like(sep, np.inf), where=dt > 0)
    dmag = np.abs(mag[j] - mag[i])
    ok = (expo[i] != expo[j]) & (dt >= cfg.min_dt_hr) & (sep >= cfg.min_sep)
    ok &= (rate >= cfg.min_rate) & (rate <= cfg.max_rate)
    ok &= dmag <= np.where(filt[i] == filt[j], cfg.max_dmag_same, cfg.max_dmag_diff)
    return np.column_stack([i[ok], j[ok]])


def _extend_pairs(pairs, xy, t, expo, mag, cfg: Config):
    """For every pair, look in each *other* exposure for a detection on the extrapolated line.

    Vectorised over pairs: one KD-tree query per exposure. Returns a set of member tuples.
    """
    i, j = pairs[:, 0], pairs[:, 1]
    v = (xy[j] - xy[i]) / (t[j] - t[i])[:, None]
    members = [[a, b] for a, b in pairs]
    for e in np.unique(expo):
        idx = np.flatnonzero(expo == e)
        te = t[idx].mean()
        pred = xy[i] + v * (te - t[i])[:, None]
        dist, k = cKDTree(xy[idx]).query(pred, distance_upper_bound=cfg.link_tol)
        hit = np.isfinite(dist) & (expo[i] != e) & (expo[j] != e)
        for p in np.flatnonzero(hit):
            m = idx[k[p]]
            if abs(mag[m] - mag[i[p]]) <= cfg.max_dmag_diff:
                members[p].append(m)
    return {tuple(sorted(m)) for m in members}


def _drop_subsets(sets):
    """Remove member sets strictly contained in a larger one (e.g. triplets inside a quad)."""
    keep, by_det = [], {}
    for s in sorted(sets, key=len, reverse=True):
        cands = set.intersection(*(by_det.get(d, set()) for d in s)) if all(d in by_det for d in s) else set()
        if cands:
            continue
        k = len(keep)
        keep.append(s)
        for d in s:
            by_det.setdefault(d, set()).add(k)
    return keep


def link_night(night: int, df: pd.DataFrame, cfg: Config, ecl_pa: float) -> list[dict]:
    """Link one night's orphans into tracklets. `df` must hold x, y [arcsec] and mjd."""
    expo = pd.factorize(df["exposure"])[0]
    if len(np.unique(expo)) < max(2, cfg.min_det):
        return []
    xy = df[["x", "y"]].to_numpy(float)
    mjd = df["mjd"].to_numpy(float)
    t = (mjd - mjd.min()) * 24.0                       # hours
    mag = df["mag_auto"].to_numpy(float)
    filt = pd.factorize(df["filter"])[0]
    ids = df["id"].to_numpy()

    pairs = _candidate_pairs(xy, t, expo, mag, filt, cfg)
    if len(pairs) == 0:
        return []
    sets = [s for s in _extend_pairs(pairs, xy, t, expo, mag, cfg) if len(s) >= cfg.min_det]

    out = []
    for s in _drop_subsets(sets):
        m = np.array(s)
        tm = t[m].mean()
        dt = t[m] - tm
        # Unweighted linear least squares: position = p0 + v * (t - t_mean)
        v = (dt[:, None] * (xy[m] - xy[m].mean(0))).sum(0) / (dt ** 2).sum()
        p0 = xy[m].mean(0)
        resid = xy[m] - (p0 + dt[:, None] * v)
        rms = float(np.sqrt((resid ** 2).sum(1).mean()))
        rate = float(np.hypot(*v))
        if rms > cfg.link_tol or not (cfg.min_rate <= rate <= cfg.max_rate):
            continue
        pa = float(np.degrees(np.arctan2(v[0], v[1])) % 360.0)
        ecl = abs((pa - ecl_pa + 90.0) % 180.0 - 90.0)   # 0 = along ecliptic, 90 = perpendicular
        if cfg.max_ecl_angle is not None and ecl > cfg.max_ecl_angle:
            continue
        ra, dec = inverse_tangent_plane(p0[0], p0[1], cfg.ra0, cfg.dec0)
        out.append(dict(
            night=int(night), n_det=len(m), arc_hr=float(np.ptp(t[m])),
            mjd_ref=float(mjd.min() + tm / 24.0), ra_ref=float(ra), dec_ref=float(dec),
            dra_dt=float(v[0]), ddec_dt=float(v[1]), rate=rate, pa_deg=pa,
            ecl_angle_deg=float(ecl), rms=rms, mag_mean=float(np.nanmean(mag[m])),
            filters="".join(sorted(set(df["filter"].to_numpy()[m].astype(str)))),
            det_ids=[str(x) for x in ids[m]],
        ))
    return out


def _worker(args):
    return link_night(*args)


def run(cfg: Config, orphans: pd.DataFrame | None = None) -> pd.DataFrame:
    if orphans is None:
        f = cfg.path("orphans.parquet")
        if not f.exists():
            raise SystemExit(f"{f} not found - Stage 1 has not completed for this field yet.")
        orphans = pd.read_parquet(f)
    ecl_pa = ecliptic_position_angle(cfg.ra0, cfg.dec0)
    jobs = [(n, g.reset_index(drop=True), cfg, ecl_pa) for n, g in orphans.groupby("night")]
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as pool:
            results = list(pool.map(_worker, jobs, chunksize=4))
    else:
        results = [_worker(j) for j in jobs]
    rows = [r for res in results for r in res]
    tr = pd.DataFrame(rows)
    if not tr.empty:
        tr = tr.sort_values(["night", "ra_ref"]).reset_index(drop=True)
        tr.insert(0, "tracklet_id", [f"{cfg.tag}_{n}_{k:05d}" for k, n in enumerate(tr["night"])])
    tr.to_parquet(cfg.path("tracklets.parquet"), index=False)
    log.info("%d tracklets from %d nights -> %s", len(tr), len(jobs), cfg.path("tracklets.parquet"))
    return tr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, default=Config.ra0)
    p.add_argument("--dec", type=float, default=Config.dec0)
    p.add_argument("--outdir", default="tno_data")
    p.add_argument("--min-rate", type=float, default=Config.min_rate)
    p.add_argument("--max-rate", type=float, default=Config.max_rate)
    p.add_argument("--min-det", type=int, default=Config.min_det, choices=[2, 3, 4])
    p.add_argument("--link-tol", type=float, default=Config.link_tol)
    p.add_argument("--max-ecl-angle", type=float, default=None)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config(ra0=a.ra, dec0=a.dec, outdir=a.outdir, min_rate=a.min_rate, max_rate=a.max_rate,
                 min_det=a.min_det, link_tol=a.link_tol, max_ecl_angle=a.max_ecl_angle, workers=a.workers)
    tr = run(cfg)
    if not tr.empty:
        print(tr.drop(columns="det_ids").to_string(max_rows=40))


if __name__ == "__main__":
    main()
