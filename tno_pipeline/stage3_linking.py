"""Stage 3 - HelioLinC-style multi-night orbit linking of orphan detections.

Why this is not the tracklet linker originally sketched
-------------------------------------------------------
A TNO at 40 AU drifts ~1.4 deg/yr across the sky, so it leaves a 1-deg field in
under a year; only objects beyond ~80 AU linger for several years. Linking the same
object across a multi-year baseline therefore means (a) pooling several fields that
follow its track, and (b) linking individual detections, because the observing
cadence rarely gives tracklets. This stage does the detection-level linking.

Method (de-parallax + windowed clustering)
------------------------------------------
For a trial heliocentric distance d, every detection is shifted by the observer's
tangential position divided by d (the parallax factor). In this heliocentric frame a
real object at distance d moves on a straight line, phi(t) = phi0 + v*(t - t0), while
foreground asteroids and wrong-distance sources scatter. Within a short time window
(so the parallax curvature left by a slightly wrong d stays small) every pair of
detections implies a (phi0, v); pairs from one object pile into the same cell of a
(phi0, v) grid. A cell holding >= min_det detections from >= min_nights nights is a
candidate arc. Candidates are merged across overlapping windows and distances, then
refit for the distance that minimises the straight-line residual.

This recovers objects seen on several nights within one field's residence window
(the night clusters in the data). Following an object across the full baseline and
across fields is the next layer, built once a strip of fields exists.

Usage
-----
    python -m tno_pipeline.stage3_linking --ra 2.0 --dec 0.0
    python -m tno_pipeline.stage3_linking --fields tno_data/ra*          # pool many fields
    python -m tno_pipeline.stage3_linking --ra 2.0 --dec 0.0 --inject    # add a synthetic TNO

Reads one or more <field>/orphans.parquet; writes <first field>/candidates.parquet.
"""
from __future__ import annotations

import argparse
import glob
import logging

import numpy as np
import pandas as pd

from .common import (ARCSEC_PER_RAD, Config, circular_sky_rate, observer_tangential,
                     tangent_plane)

log = logging.getLogger("stage3")

# A real point source sits at the seeing; below ~0.7\" is a cosmic ray/hot pixel, above
# ~2.2\" is an extended source (galaxy). Detections outside this band are not linkable TNOs.
MIN_FWHM = 0.7
MAX_FWHM = 2.2

# numba is optional; it is not used by the KD-tree pair finder below, but kept available
# for anyone extending the kernel. HAVE_NUMBA just records whether it imported.
try:
    import numba          # noqa: F401
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def _pairs_in_window(phi_e, phi_n, t, expo, t0, window, min_dt, vmax):
    """Detection pairs implying a bound-orbit velocity, within one window.

    A cKDTree restricts pairs to those whose de-parallaxed separation is small enough to
    be reached at <= vmax over the window, so cost scales with the number of nearby pairs
    rather than N^2. Returns (th0e, th0n, ve, vn, ia, ib) as numpy arrays indexing the
    window arrays.
    """
    from scipy.spatial import cKDTree
    xy = np.column_stack([phi_e, phi_n])
    pr = cKDTree(xy).query_pairs(vmax * window, output_type="ndarray")
    if len(pr) == 0:
        z = np.empty(0)
        return z, z, z, z, z.astype(np.int64), z.astype(np.int64)
    a, b = pr[:, 0], pr[:, 1]
    swap = t[b] < t[a]                                   # order each pair in time
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    dt = t[b] - t[a]
    good = dt >= max(min_dt, 1e-9)
    dt_safe = np.where(good, dt, 1.0)
    ve = (phi_e[b] - phi_e[a]) / dt_safe
    vn = (phi_n[b] - phi_n[a]) / dt_safe
    ok = good & (dt <= window) & (expo[a] != expo[b]) & (ve * ve + vn * vn <= vmax * vmax)
    a, b, ve, vn = a[ok], b[ok], ve[ok], vn[ok]
    th0e = phi_e[a] - ve * (t[a] - t0)
    th0n = phi_n[a] - vn * (t[a] - t0)
    return th0e, th0n, ve, vn, a, b


# ----------------------------------------------------------------------------
def collapse_nightly(df, radius=1.5):
    """Collapse each night's detections of one source (multi-filter bursts, same position
    within `radius` arcsec) into a single nightly point. This stops a 3-night link from
    being reported as "7 detections" and lets us test brightness consistency per filter.

    Returns a reduced frame with one row per (night, source), carrying `members` (the
    original detection ids) and `mags` (list of (filter, mag_auto)) for later checks.
    """
    from scipy.spatial import cKDTree
    rows = []
    for night, g in df.groupby("night"):
        g = g.reset_index(drop=True)
        xy = g[["x", "y"]].to_numpy(float)
        if len(g) == 1:
            comp, ncomp = np.array([0]), 1
        else:
            # connected components under a radius graph = one source's multi-filter burst
            pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
            parent = np.arange(len(g))
            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]; i = parent[i]
                return i
            for a, b in pairs:
                parent[find(a)] = find(b)
            comp = np.array([find(i) for i in range(len(g))])
            _, comp = np.unique(comp, return_inverse=True); ncomp = comp.max() + 1
        for c in range(ncomp):
            sel = g.iloc[np.flatnonzero(comp == c)]
            best = sel.loc[sel["magerr_auto"].idxmin()]            # representative = best-SNR detection
            rows.append(dict(
                id=f"N{int(night)}_{c}", objectid=best["objectid"], exposure=f"night{int(night)}",
                ccdnum=best["ccdnum"], filter=best["filter"], mjd=float(sel["mjd"].mean()),
                ra=float(sel["ra"].mean()), dec=float(sel["dec"].mean()),
                mag_auto=float(best["mag_auto"]), magerr_auto=float(best["magerr_auto"]),
                fwhm=float(sel["fwhm"].mean()), class_star=float(sel["class_star"].max()),
                night=int(night), x=float(sel["x"].mean()), y=float(sel["y"].mean()),
                b_e=float(sel["b_e"].mean()), b_n=float(sel["b_n"].mean()),
                members=[str(v) for v in sel["id"]],
                mags=[(str(f), float(m)) for f, m in zip(sel["filter"], sel["mag_auto"])]))
    out = pd.DataFrame(rows)
    return out


def brightness_ok(members_mags, max_dmag):
    """True if, in every filter seen on >=2 nights, the magnitude spread is < max_dmag.
    A real moving object holds roughly constant brightness per band; a chance alignment
    that strings together sources of very different brightness does not.
    """
    by_filt = {}
    for per_night in members_mags:
        seen = {}
        for f, m in per_night:
            seen.setdefault(f, []).append(m)
        for f, ms in seen.items():
            by_filt.setdefault(f, []).append(min(ms))     # one (brightest) value per night
    for f, vals in by_filt.items():
        if len(vals) >= 2 and (max(vals) - min(vals)) > max_dmag:
            return False
    return True


# ----------------------------------------------------------------------------
def load_pool(paths, ra0, dec0):
    """Concatenate orphan tables and attach tangent-plane x,y at a common centre and
    the per-detection parallax factors b_e, b_n (AU)."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["field"] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id").reset_index(drop=True)
    n0 = len(df)
    df = df[(df["fwhm"] >= MIN_FWHM) & (df["fwhm"] <= MAX_FWHM)].reset_index(drop=True)
    log.info("PSF cut: dropped %d sub-seeing/extended detections (kept %d)", n0 - len(df), len(df))
    df["x"], df["y"] = tangent_plane(df["ra"].values, df["dec"].values, ra0, dec0)
    df["b_e"], df["b_n"] = observer_tangential(df["mjd"].values, ra0, dec0)
    return df


def link(df: pd.DataFrame, cfg: Config, dists, window_days, step_days,
         vprior_mult, pos_cell, vel_cells, min_det, min_nights, rms_tol,
         window_scale=1.5, max_window=45.0):
    """Core linker. Returns a list of candidate dicts.

    window_days/step_days may be None, in which case the window is set per distance to
    window_scale x the time an object at that distance stays in the field (size / rate),
    capped at max_window: far objects linger and need long windows, near ones only short.
    """
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)
    be = df["b_e"].to_numpy(float); bn = df["b_n"].to_numpy(float)
    t = df["mjd"].to_numpy(float)
    night = df["night"].to_numpy(np.int64)
    expo = pd.factorize(df["exposure"])[0].astype(np.int64)
    gid = np.arange(len(df))

    size_arcsec = cfg.size_deg * 3600.0
    baseline = float(t.max() - t.min())
    seeds = []                    # list of (member-index-array) harvested per distance
    import time as _time
    t_start = _time.time()
    for di, d in enumerate(dists):
        phi_e = x + (be / d) * ARCSEC_PER_RAD
        phi_n = y + (bn / d) * ARCSEC_PER_RAD
        vmax = vprior_mult * circular_sky_rate(d)                 # arcsec/day
        vcell = max(vmax / vel_cells, 1e-6)
        win = window_days if window_days else min(
            window_scale * size_arcsec / circular_sky_rate(d), max_window, baseline + 1)
        stp = step_days if step_days else max(win / 3.0, 1.0)
        cells: dict = {}          # cell key -> set of global det ids, for THIS distance only
        for s0 in np.arange(t.min(), t.max() + 1, stp):
            m = (t >= s0) & (t < s0 + win)
            if m.sum() < min_det or np.unique(night[m]).size < min_nights:
                continue
            idx = np.flatnonzero(m)
            idx = idx[np.argsort(t[idx], kind="stable")]
            t0 = t[idx].mean()
            th0e, th0n, ve, vn, ia, ib = _pairs_in_window(
                phi_e[idx], phi_n[idx], t[idx], expo[idx],
                float(t0), float(win), float(cfg.min_dt_hr / 24.0), float(vmax))
            ke = np.round(th0e / pos_cell).astype(np.int64)
            kn = np.round(th0n / pos_cell).astype(np.int64)
            kve = np.round(ve / vcell).astype(np.int64)
            kvn = np.round(vn / vcell).astype(np.int64)
            ga, gb = idx[ia], idx[ib]
            for p_ in range(ia.shape[0]):
                key = (ke[p_], kn[p_], kve[p_], kvn[p_])
                st = cells.get(key)
                if st is None:
                    st = set(); cells[key] = st
                st.add(int(ga[p_])); st.add(int(gb[p_]))
        # harvest this distance's qualifying cells, then drop the dict (bounds memory)
        for dets in cells.values():
            if len(dets) >= min_det and np.unique(night[list(dets)]).size >= min_nights:
                seeds.append(np.array(sorted(dets)))
        del cells
        if (di + 1) % 10 == 0 or di == len(dists) - 1:
            log.info("  distance %d/%d (%.0f AU): %d seeds so far, %.0fs",
                     di + 1, len(dists), d, len(seeds), _time.time() - t_start)

    # Each harvested cell is a seed (a set of detections). Evaluate every seed on its own,
    # then drop duplicates: the same object recurs across distances and overlapping windows,
    # producing many seeds that share most of their detections.
    if not seeds:
        return []
    log.info("refitting %d seeds", len(seeds))

    scored = []
    for m in seeds:
        best = _refit(x[m], y[m], be[m], bn[m], t[m], dists)
        if best is None or best["rms"] > rms_tol:
            continue
        best.update(members=m, n_det=int(m.size), n_nights=int(np.unique(night[m]).size))
        scored.append(best)
    # best first: most nights, then most detections, then tightest fit
    scored.sort(key=lambda c: (-c["n_nights"], -c["n_det"], c["rms"]))

    out, claimed = [], []
    for c in scored:
        ms = set(int(v) for v in c["members"])
        if any(len(ms & other) >= 2 for other in claimed):   # same object already kept
            continue
        claimed.append(ms)
        m = c.pop("members")
        c.update(mjd_min=float(t[m].min()), mjd_max=float(t[m].max()),
                 arc_days=float(np.ptp(t[m])), mag_mean=float(np.nanmean(df["mag_auto"].values[m])),
                 filters="".join(sorted(set(df["filter"].astype(str).values[m]))),
                 det_ids=[str(v) for v in df["id"].values[m]])
        out.append(c)
    return out


def _refit(x, y, be, bn, t, dists):
    """Fit the straight-line heliocentric model at each trial distance and keep the best.

    Distance is constrained only by the curvature of Earth's parallax over the arc, so a
    short arc fits well across a wide distance range. We therefore also report the range
    of distances whose residual is within sqrt(2) of the minimum - a wide range means the
    arc does not pin the distance (expected for arcs spanning much less than a year).
    """
    t0 = t.mean(); dt = t - t0
    A = np.column_stack([np.ones_like(dt), dt])            # phi = c0 + c1*dt
    rmss = np.empty(len(dists))
    fits = []
    for i, d in enumerate(dists):
        pe = x + (be / d) * ARCSEC_PER_RAD
        pn = y + (bn / d) * ARCSEC_PER_RAD
        ce, *_ = np.linalg.lstsq(A, pe, rcond=None)
        cn, *_ = np.linalg.lstsq(A, pn, rcond=None)
        res = np.hypot(pe - A @ ce, pn - A @ cn)
        rmss[i] = np.sqrt((res ** 2).mean())
        fits.append((ce, cn))
    i = int(np.argmin(rmss))
    ce, cn = fits[i]
    within = np.flatnonzero(rmss <= max(rmss[i] * np.sqrt(2.0), rmss[i] + 0.3))
    return dict(dist_au=float(dists[i]), dist_lo=float(dists[within.min()]),
                dist_hi=float(dists[within.max()]), rms=float(rmss[i]), mjd_ref=float(t0),
                phi0_e=float(ce[0]), phi0_n=float(cn[0]), v_e=float(ce[1]), v_n=float(cn[1]))


def run(cfg: Config, paths, inject=False, **kw):
    df = load_pool(paths, cfg.ra0, cfg.dec0)
    log.info("pooled %d orphan detections from %d field(s), %d nights, MJD %.0f-%.0f",
             len(df), len(paths), df["night"].nunique(), df["mjd"].min(), df["mjd"].max())
    if inject:
        df = pd.concat([df, _synth_object(cfg, df)], ignore_index=True)
        log.info("injected a synthetic object; now %d detections", len(df))

    members, mags = None, None
    if kw.get("collapse", True):
        work = collapse_nightly(df, kw["collapse_radius"])
        members = dict(zip(work["id"], work["members"]))
        mags = dict(zip(work["id"], work["mags"]))
        log.info("collapsed to %d nightly sources (from %d detections)", len(work), len(df))
    else:
        work = df
        work["members"] = [[v] for v in work["id"]]
        members = dict(zip(work["id"], work["members"]))
        mags = {r.id: [(r.filter, r.mag_auto)] for r in work.itertuples()}

    dmin, dmax, dstep = kw["dmin"], kw["dmax"], kw["dstep"]
    dists = np.arange(dmin, dmax + 1e-9, dstep)
    cands = link(work, cfg, dists, kw["window"], kw["step"], kw["vprior"],
                 kw["pos_cell"], kw["vel_cells"], kw["min_det"], kw["min_nights"],
                 kw["rms_tol"], window_scale=kw["window_scale"], max_window=kw["max_window"])

    from .common import inverse_tangent_plane
    kept = []
    for c in cands:
        nightly_ids = c["det_ids"]                               # these are nightly-source ids
        per_night_mags = [mags[nid] for nid in nightly_ids]
        if kw.get("collapse", True) and not brightness_ok(per_night_mags, kw["max_dmag"]):
            continue                                             # brightness jumps between nights
        c["ra_ref"], c["dec_ref"] = inverse_tangent_plane(c["phi0_e"], c["phi0_n"], cfg.ra0, cfg.dec0)
        c["n_obs"] = int(sum(len(members[nid]) for nid in nightly_ids))
        real_ids = [m for nid in nightly_ids for m in members[nid]]
        c["det_ids"] = real_ids
        # --- quality grade -------------------------------------------------
        rate = float(np.hypot(c["v_e"], c["v_n"]))                 # de-parallaxed sky motion [arcsec/day]
        circ = circular_sky_rate(c["dist_au"])
        c["true_rate"] = round(rate, 3)
        c["bound_ok"] = bool(rate <= kw.get("bound_factor", 1.5) * circ)
        # strongest single filter: nights seen and brightness spread in it
        sm = df.loc[df["id"].isin(real_ids), ["filter", "mjd", "mag_auto"]].copy()
        sm["ngt"] = np.floor(sm["mjd"] - cfg.night_offset).astype(int)
        nf_n, nf_spread, nf_filter = 0, np.nan, ""
        for f, gg in sm.groupby("filter"):
            per = gg.groupby("ngt")["mag_auto"].min()
            if len(per) > nf_n:
                nf_n, nf_spread, nf_filter = len(per), float(per.max() - per.min()), str(f)
        c["filt_nights"], c["filt_spread"], c["best_filter"] = nf_n, round(nf_spread, 3) if nf_n else np.nan, nf_filter
        fw = df.loc[df["id"].isin(real_ids), "fwhm"].values
        psf_ok = (len(fw) > 0 and fw.min() >= MIN_FWHM and fw.max() <= MAX_FWHM
                  and fw.max() / max(fw.min(), 1e-3) <= 1.8)
        c["fwhm_min"], c["fwhm_max"], c["psf_ok"] = round(float(fw.min()), 2), round(float(fw.max()), 2), bool(psf_ok)
        c["grade"] = ("A" if (c["bound_ok"] and psf_ok and nf_n >= 3 and nf_spread <= 0.6)
                      else "B" if (c["bound_ok"] and psf_ok and nf_n >= 2 and (np.isnan(nf_spread) or nf_spread <= 1.0))
                      else "C")
        kept.append(c)
    log.info("%d candidates after brightness-consistency cut (from %d)", len(kept), len(cands))

    cols = ["grade", "dist_au", "dist_lo", "dist_hi", "n_nights", "n_obs", "arc_days",
            "mjd_ref", "ra_ref", "dec_ref", "true_rate", "bound_ok", "psf_ok",
            "fwhm_min", "fwhm_max", "best_filter", "filt_nights", "filt_spread", "rms",
            "mag_mean", "filters", "mjd_min", "mjd_max", "det_ids"]
    cand = pd.DataFrame(kept)
    if not cand.empty:
        grank = {"A": 0, "B": 1, "C": 2}
        cand["_g"] = cand["grade"].map(grank)
        cand = cand.sort_values(["_g", "filt_nights", "rms"], ascending=[True, False, True])
        cand = cand[[c for c in cols if c in cand.columns]]
    out = Config(ra0=cfg.ra0, dec0=cfg.dec0, outdir=cfg.outdir).path("candidates.parquet")
    cand.to_parquet(out, index=False)
    log.info("%d candidate orbit arc(s) -> %s", len(cand), out)
    return cand


# ----------------------------------------------------------------------------
def _synth_object(cfg: Config, df: pd.DataFrame, dist=42.0, seed=7):
    """Inject one distant object: pick the richest cluster of nights, place a source at
    `dist` AU moving on a straight heliocentric line, observed on those nights with the
    correct parallax signature and 0.08 arcsec noise."""
    rng = np.random.default_rng(seed)
    nights = df.groupby("night")["mjd"].min().sort_values()
    # richest 3-night cluster within 20 days
    arr = nights.values
    best = None
    for i in range(len(arr)):
        grp = arr[(arr >= arr[i]) & (arr < arr[i] + 20)]
        if len(grp) >= 3 and (best is None or len(grp) > len(best)):
            best = grp
    if best is None:
        best = arr[:3]
    # a true heliocentric linear track: pick phi0 near field centre, small bound velocity
    t0 = best.mean()
    v = rng.normal(0, 0.3 * circular_sky_rate(dist), 2)     # arcsec/day
    phi0 = rng.uniform(-200, 200, 2)
    rows = []
    for mj in best:
        be, bn = observer_tangential(np.array([mj]), cfg.ra0, cfg.dec0)
        phi = phi0 + v * (mj - t0)
        x = phi[0] - (be[0] / dist) * ARCSEC_PER_RAD + rng.normal(0, 0.08)
        y = phi[1] - (bn[0] / dist) * ARCSEC_PER_RAD + rng.normal(0, 0.08)
        from .common import inverse_tangent_plane
        ra, dec = inverse_tangent_plane(x, y, cfg.ra0, cfg.dec0)
        rows.append(dict(id=f"SYNTH.{mj:.3f}", objectid="", exposure=f"synth_{mj:.3f}", ccdnum=1,
                         filter="r", mjd=float(mj), ra=float(ra), dec=float(dec), raerr=0.05,
                         decerr=0.05, mag_auto=23.0, magerr_auto=0.1, fwhm=1.1, class_star=0.5,
                         flags=0, night=int(np.floor(mj - 0.7)), x=float(x), y=float(y),
                         b_e=float(be[0]), b_n=float(bn[0]), field="synthetic"))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, default=Config.ra0)
    p.add_argument("--dec", type=float, default=Config.dec0)
    p.add_argument("--outdir", default="tno_data")
    p.add_argument("--fields", nargs="*", help="glob(s) of field dirs to pool; default is the one field")
    p.add_argument("--dmin", type=float, default=35.0)
    p.add_argument("--dmax", type=float, default=120.0)
    p.add_argument("--dstep", type=float, default=2.0, help="distance grid step [AU]")
    p.add_argument("--window", type=float, default=None,
                   help="fixed linking window [days]; default auto-scales with distance")
    p.add_argument("--step", type=float, default=None, help="fixed window step [days]")
    p.add_argument("--window-scale", type=float, default=1.5,
                   help="auto window = scale x field residence time at each distance")
    p.add_argument("--max-window", type=float, default=45.0,
                   help="cap on the auto window [days]. Links detections within this span; "
                        "raise it (slower) to reach objects seen across a longer baseline")
    p.add_argument("--vprior", type=float, default=2.0, help="velocity prior, x circular rate")
    p.add_argument("--pos-cell", type=float, default=3.0, help="position cell [arcsec]")
    p.add_argument("--vel-cells", type=float, default=8.0, help="velocity cells across the prior")
    p.add_argument("--min-det", type=int, default=3)
    p.add_argument("--min-nights", type=int, default=3)
    p.add_argument("--rms-tol", type=float, default=0.5,
                   help="max straight-line residual [arcsec]; raise for noisier astrometry")
    p.add_argument("--collapse-radius", type=float, default=1.5,
                   help="merge a night's detections within this radius [arcsec] into one point")
    p.add_argument("--no-collapse", action="store_true", help="disable nightly collapse")
    p.add_argument("--max-dmag", type=float, default=1.0,
                   help="reject a link whose per-filter brightness varies by more than this")
    p.add_argument("--bound-factor", type=float, default=1.5,
                   help="grade A/B require de-parallax motion <= this x circular-orbit rate")
    p.add_argument("--inject", action="store_true", help="add one synthetic TNO and check recovery")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config(ra0=a.ra, dec0=a.dec, outdir=a.outdir)
    if a.fields:
        paths = sorted({f for g in a.fields for f in glob.glob(g.rstrip("/") + "/orphans.parquet")}
                       | {f for g in a.fields for f in glob.glob(g.rstrip("/") + "/*/orphans.parquet")})
    else:
        paths = [str(cfg.path("orphans.parquet"))]
    if not paths:
        raise SystemExit("no orphans.parquet found for the given field(s)")
    cand = run(cfg, paths, inject=a.inject, dmin=a.dmin, dmax=a.dmax, dstep=a.dstep,
               window=a.window, step=a.step, window_scale=a.window_scale, vprior=a.vprior,
               pos_cell=a.pos_cell, max_window=a.max_window,
               vel_cells=a.vel_cells, min_det=a.min_det, min_nights=a.min_nights, rms_tol=a.rms_tol,
               collapse=not a.no_collapse, collapse_radius=a.collapse_radius, max_dmag=a.max_dmag,
               bound_factor=a.bound_factor)
    if not cand.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(cand.drop(columns="det_ids").round(3).to_string())


if __name__ == "__main__":
    main()
