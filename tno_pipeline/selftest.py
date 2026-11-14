"""Offline end-to-end test of Stages 1-2 on synthetic data (no network needed).

    python -m tno_pipeline.selftest

Builds a 0.2 x 0.2 deg field with stationary sources seen on many nights, random
single-epoch junk, and three injected movers, then checks that the stationary
filter keeps the movers and that the linker recovers their rates.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from . import stage1_ingest, stage2_tracklets
from .common import Config, inverse_tangent_plane


def make_synthetic(cfg: Config, seed: int = 1):
    rng = np.random.default_rng(seed)
    half = 360.0                                   # arcsec
    nights = [57000.0, 57001.0, 57030.0, 57380.0, 57745.0, 58100.0]
    offsets = [0.05, 0.10, 0.16, 0.22]             # exposure times within a night [d], +1.0 UT
    stars = rng.uniform(-half, half, (3000, 2))
    star_mag = rng.uniform(17, 23.5, len(stars))
    movers = [  # (night index, x0, y0, vx, vy) in arcsec and arcsec/hr
        (1, -200.0, 50.0, -2.9, -0.8), (3, 120.0, -150.0, -1.2, 0.3), (4, 10.0, 250.0, 3.6, 1.1)]
    rows, n = [], 0
    for ni, mjd0 in enumerate(nights):
        for ei, off in enumerate(offsets):
            mjd, expo = mjd0 + 1.0 + off, f"c4d_{ni:02d}_{ei:02d}"
            seen = rng.random(len(stars)) < 0.9
            pos = stars[seen] + rng.normal(0, 0.08, (seen.sum(), 2))
            junk = rng.uniform(-half, half, (80, 2))
            blocks = [(pos, star_mag[seen], "s"), (junk, rng.uniform(21, 24, len(junk)), "j")]
            for mi, (mn, x0, y0, vx, vy) in enumerate(movers):
                if mn == ni:
                    p = np.array([[x0 + vx * off * 24, y0 + vy * off * 24]]) + rng.normal(0, 0.08, (1, 2))
                    blocks.append((p, np.array([23.0 + 0.1 * rng.normal()]), f"m{mi}"))
            for p, m, kind in blocks:
                ra, dec = inverse_tangent_plane(p[:, 0], p[:, 1], cfg.ra0, cfg.dec0)
                rows.append(pd.DataFrame(dict(
                    measid=[f"{expo}.{kind}.{n + k}" for k in range(len(p))], objectid="", exposure=expo,
                    ccdnum=1, filter="r", mjd=mjd, ra=ra, dec=dec, raerr=0.05, decerr=0.05,
                    mag_auto=m, magerr_auto=0.1, fwhm=1.1, class_star=0.8, flags=0)))
                n += len(p)
    return pd.concat(rows, ignore_index=True), movers


def main():
    cfg = Config(ra0=2.0, dec0=0.0, outdir=tempfile.mkdtemp())
    raw, movers = make_synthetic(cfg)
    det = stage1_ingest.find_orphans(stage1_ingest.quality_cuts(raw, cfg), cfg).rename(columns={"measid": "id"})
    is_star = det["id"].str.contains(".s.", regex=False)
    is_mover = det["id"].str.contains(".m", regex=False)
    print(f"detections: {len(det)}  stars flagged stationary: {det.loc[is_star, 'stationary'].mean():.4f}"
          f"  movers kept: {(~det.loc[is_mover, 'stationary']).sum()}/{is_mover.sum()}")
    assert det.loc[is_star, "stationary"].mean() > 0.999
    # A mover detection that lands within 1.5" of a star is (correctly) lost to the
    # stationary filter; with ~4% of this crowded field masked, expect the odd casualty.
    assert (~det.loc[is_mover, "stationary"]).sum() >= is_mover.sum() - 2

    tr = stage2_tracklets.run(cfg, det[~det["stationary"]].reset_index(drop=True))
    print(tr.drop(columns=["det_ids", "tracklet_id"]).round(3).to_string())
    for mi, (_, _, _, vx, vy) in enumerate(movers):
        hit = tr[tr["det_ids"].map(lambda ids: sum(f".m{mi}." in s for s in ids) >= 3)]
        assert len(hit) == 1, f"mover {mi} not recovered"
        r = hit.iloc[0]
        assert abs(r.dra_dt - vx) < 0.15 and abs(r.ddec_dt - vy) < 0.15, f"mover {mi} rate wrong"
    n_false = len(tr) - len(movers)
    print(f"recovered {len(movers)}/{len(movers)} injected movers; {n_false} false tracklet(s)")


if __name__ == "__main__":
    main()
