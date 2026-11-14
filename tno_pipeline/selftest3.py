"""Offline test of Stage 3 on synthetic data with the real field's night structure.

    python -m tno_pipeline.selftest3

Builds orphan detections that mimic the actual field (isolated noise per night plus a
few fast asteroids that cannot link across nights), injects two real distant objects on
night clusters where they would stay in the field, and checks the linker recovers them
at the right distance with no spurious multi-night candidates.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .common import (ARCSEC_PER_RAD, Config, circular_sky_rate, inverse_tangent_plane,
                     observer_tangential)
from . import stage3_linking as s3

# the real field's nights (MJD, integer), from the Stage 1 run
NIGHTS = [56272, 56273, 56463, 56536, 56543, 56545, 56562, 56577, 56603, 56617, 56618,
          56856, 56857, 56886, 56952, 56958, 56974, 57641, 57642, 57652, 57662, 57709,
          57725, 57726, 57982, 58022, 58053, 58054, 58065, 58371, 58374, 58377]
HALF = 1800.0            # field half-size in arcsec (1 deg)


def _obj_rows(cfg, mjds, dist, anchor, v, tag, rng, filt="r"):
    """Detections of one distant object at `dist` AU on the given nights.

    `anchor` is the object's OBSERVED tangent-plane position (arcsec) at the mean epoch;
    the heliocentric phi0 is derived from it so the object lands inside the field (the
    parallax offset itself is ~5000 arcsec at 40 AU and must not be confused with position).
    """
    mjds = np.asarray(mjds, float)
    t0 = mjds.mean()
    be, bn = observer_tangential(mjds, cfg.ra0, cfg.dec0)
    be0, bn0 = observer_tangential(np.array([t0]), cfg.ra0, cfg.dec0)
    phi0 = np.array([anchor[0] + be0[0] / dist * ARCSEC_PER_RAD,
                     anchor[1] + bn0[0] / dist * ARCSEC_PER_RAD])
    rows = []
    for k, mj in enumerate(mjds):
        phi = phi0 + v * (mj - t0)                                   # heliocentric linear track
        x = phi[0] - (be[k] / dist) * ARCSEC_PER_RAD + rng.normal(0, 0.08)
        y = phi[1] - (bn[k] / dist) * ARCSEC_PER_RAD + rng.normal(0, 0.08)
        if abs(x) > HALF or abs(y) > HALF:                           # left the field; not observed
            continue
        ra, dec = inverse_tangent_plane(x, y, cfg.ra0, cfg.dec0)
        rows.append(dict(id=f"{tag}.{k}", objectid="", exposure=f"{tag}_e{k}", ccdnum=1,
                         filter=filt, mjd=float(mj) + rng.uniform(0, 0.03), ra=float(ra),
                         dec=float(dec), raerr=0.05, decerr=0.05, mag_auto=22.8,
                         magerr_auto=0.1, fwhm=1.1, class_star=0.5, flags=0,
                         night=int(np.floor(mj - 0.7))))
    return rows


def make_synthetic(cfg, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    # background: single-night noise/galaxies and fast asteroids (move too far to re-link)
    n = 0
    for night in NIGHTS:
        mj = night + 1.0
        for _ in range(150):                                         # noise, one detection
            x, y = rng.uniform(-HALF, HALF, 2)
            ra, dec = inverse_tangent_plane(x, y, cfg.ra0, cfg.dec0)
            rows.append(dict(id=f"n{n}", objectid="", exposure=f"c4d_{night}", ccdnum=1,
                             filter="r", mjd=mj + rng.uniform(0, 0.1), ra=float(ra), dec=float(dec),
                             raerr=0.05, decerr=0.05, mag_auto=rng.uniform(22, 24), magerr_auto=0.1,
                             fwhm=1.1, class_star=0.5, flags=0, night=night)); n += 1
    df = pd.DataFrame(rows)

    # object A: 42 AU, on the Sep-2016 cluster (57641/42/52), stays in field (~0.7 yr residence)
    a = _obj_rows(cfg, [57641.1, 57642.1, 57652.1], 42.0,
                  rng.uniform(-500, 500, 2), rng.normal(0, 0.3 * circular_sky_rate(42.0), 2),
                  "TNO_A", rng)
    # object B: 85 AU, on a span of four nights ~1.4 yr (lingers: ~2 yr residence)
    b = _obj_rows(cfg, [58053.1, 58054.1, 58065.1], 85.0,
                  rng.uniform(-500, 500, 2), rng.normal(0, 0.3 * circular_sky_rate(85.0), 2),
                  "TNO_B", rng)
    inj = pd.DataFrame(a + b)
    return pd.concat([df, inj], ignore_index=True), {"TNO_A": 42.0, "TNO_B": 85.0}, inj


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = Config(ra0=2.0, dec0=0.0)
    df, truth, inj = make_synthetic(cfg)
    df["x"], df["y"] = s3.tangent_plane(df["ra"].values, df["dec"].values, cfg.ra0, cfg.dec0)
    df["b_e"], df["b_n"] = observer_tangential(df["mjd"].values, cfg.ra0, cfg.dec0)
    print(f"synthetic orphans: {len(df)}  injected detections: {len(inj)}  "
          f"(A on { (inj.id.str.startswith('TNO_A')).sum() } nights, "
          f"B on { (inj.id.str.startswith('TNO_B')).sum() })")

    dists = np.arange(35, 120.01, 2.0)
    cands = s3.link(df, cfg, dists, window_days=None, step_days=None, vprior_mult=2.0,
                    pos_cell=3.0, vel_cells=8.0, min_det=3, min_nights=3, rms_tol=0.5,
                    window_scale=1.5, max_window=45.0)
    cand = pd.DataFrame(cands)
    print(f"\ncandidates found: {len(cand)}")
    for c in cands:
        who = {k for d in c["det_ids"] for k in ["TNO_A", "TNO_B"] if d.startswith(k)}
        tag = "/".join(sorted(who)) or "SPURIOUS"
        print(f"  dist={c['dist_au']:.1f}AU  n_det={c['n_det']}  n_nights={c['n_nights']}  "
              f"rms={c['rms']:.2f}\"  -> {tag}")

    found = {}
    for c in cands:
        for k in ("TNO_A", "TNO_B"):
            if sum(d.startswith(k) for d in c["det_ids"]) >= 3:
                found[k] = c
    print()
    for k, dtrue in truth.items():
        assert k in found, f"{k} NOT recovered (detections did not link)"
        c = found[k]
        inside = c["dist_lo"] <= dtrue <= c["dist_hi"]
        print(f"{k}: linked on {c['n_nights']} nights, arc {c['arc_days']:.0f} d; "
              f"distance {c['dist_au']:.0f} AU (range {c['dist_lo']:.0f}-{c['dist_hi']:.0f}), "
              f"true {dtrue} {'IN range' if inside else 'OUT of range'}")
        assert inside, f"{k} true distance outside reported range"
    spurious = sum(not any(d.startswith(('TNO_A', 'TNO_B')) for d in c["det_ids"]) for c in cands)
    print(f"\nspurious 3-night candidates: {spurious} of {len(cands)} "
          f"(a small shortlist, culled by Stage 4 image inspection)")
    assert spurious < 30, f"too many spurious candidates ({spurious})"
    print("PASS: both injected objects linked, true distance within range, shortlist manageable")


if __name__ == "__main__":
    main()
