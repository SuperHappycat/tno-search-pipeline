"""Autonomous ecliptic-strip TNO search: download several fields, link, verify, report.

Runs entirely in the cloud (no local machine needed) once datalab.noirlab.edu and
vo.imcce.fr are allow-listed. Processes each field independently through Stages 1, 3 and
4 (cross-field multi-year linking is a later addition), then aggregates the graded
candidates into one report.

    python -m tno_pipeline.strip_search --ra0 2.0 --dec0 0.0 --nfields 6 --delete-tiles

Outputs under <outdir>/:
    <field tag>/...            per-field Stage 1-4 products
    strip_report.md            combined summary
    strip_candidates.parquet   all grade-A/B candidates across the strip
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Config
from . import stage1_ingest, stage3_linking, stage4_verify

log = logging.getLogger("strip")


def strip_fields(ra0, dec0, n, spacing_deg):
    """n field centres stepped along the ecliptic through (ra0, dec0), centred on the input."""
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

    ec = SkyCoord(ra0 * u.deg, dec0 * u.deg).transform_to(BarycentricMeanEcliptic())
    offs = (np.arange(n) - (n - 1) / 2.0) * spacing_deg
    out = []
    for dlon in offs:
        c = SkyCoord(ec.lon + dlon * u.deg, ec.lat, frame=BarycentricMeanEcliptic()).icrs
        out.append((round(float(c.ra.deg), 4), round(float(c.dec.deg), 4)))
    return out


def run_field(ra, dec, size, outdir, delete_tiles, do_stage4, grades):
    cfg = Config(ra0=ra, dec0=dec, size_deg=size, outdir=outdir)
    t0 = time.time()
    log.info("=== FIELD %s (ra=%.4f dec=%.4f) ===", cfg.tag, ra, dec)
    try:
        orph = stage1_ingest.run(cfg)                         # download + stationary filter
    except SystemExit as e:
        log.warning("field %s skipped: %s", cfg.tag, e); return None
    if delete_tiles:
        shutil.rmtree(cfg.path("tiles"), ignore_errors=True)  # reclaim raw download space
    cand = stage3_linking.run(cfg, [str(cfg.path("orphans.parquet"))],
                              dmin=35.0, dmax=120.0, dstep=2.0, window=None, step=None,
                              window_scale=1.5, max_window=45.0, vprior=2.0, pos_cell=3.0,
                              vel_cells=8.0, min_det=3, min_nights=3, rms_tol=0.5,
                              collapse=True, collapse_radius=1.5, max_dmag=1.0, bound_factor=1.5)
    # cross-match grade A/B against known minor planets, using the observed positions in orphans
    if not cand.empty and do_stage4:
        try:
            cand = stage4_verify.annotate_known(cand, orph, grades=("A", "B"))
        except Exception as e:
            log.warning("cross-match failed for %s: %s", cfg.tag, e)
            cand["known_object"] = ""
    elif not cand.empty:
        cand["known_object"] = ""
    # cutouts only for the real shortlist: grade-A that is NOT a known object
    if do_stage4 and not cand.empty:
        shortlist = cand[(cand["grade"] == "A") & (cand["known_object"] == "")]
        orph_ix = orph.set_index("id")
        outdir = cfg.path("cutouts"); outdir.mkdir(exist_ok=True)
        for i, c in shortlist.iterrows():
            try:
                stage4_verify.cutout_grid(c, orph_ix, outdir / f"cand_{i:04d}.png")
            except Exception as e:
                log.warning("cutout failed for %s cand %d: %s", cfg.tag, i, e)
    nA = int((cand["grade"] == "A").sum()) if not cand.empty else 0
    nA_new = int(((cand["grade"] == "A") & (cand["known_object"] == "")).sum()) if not cand.empty else 0
    log.info("field %s done in %.0f s: %d candidates, %d grade-A (%d unmatched by SkyBoT)",
             cfg.tag, time.time() - t0, len(cand), nA, nA_new)
    cand["field"] = cfg.tag
    return cand


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra0", type=float, default=2.0)
    p.add_argument("--dec0", type=float, default=0.0)
    p.add_argument("--nfields", type=int, default=6)
    p.add_argument("--spacing", type=float, default=1.0, help="field spacing along ecliptic [deg]")
    p.add_argument("--size", type=float, default=1.0, help="field size [deg]")
    p.add_argument("--outdir", default="tno_data")
    p.add_argument("--delete-tiles", action="store_true", help="delete raw tiles after each field")
    p.add_argument("--no-stage4", action="store_true", help="skip cross-match/cutouts")
    p.add_argument("--grades", default="A", help="grades to send to Stage 4")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    fields = strip_fields(a.ra0, a.dec0, a.nfields, a.spacing)
    log.info("strip of %d fields along the ecliptic: %s", len(fields), fields)
    t0 = time.time()
    allc = []
    for ra, dec in fields:
        c = run_field(ra, dec, a.size, a.outdir, a.delete_tiles, not a.no_stage4,
                      a.grades.split(","))
        if c is not None and not c.empty:
            allc.append(c)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(allc, ignore_index=True) if allc else pd.DataFrame()
    if not combined.empty:
        keep = combined[combined["grade"].isin(["A", "B"])].sort_values(
            ["grade", "n_nights", "rms"], ascending=[True, False, True])
        keep.to_parquet(out / "strip_candidates.parquet", index=False)
    # markdown report
    lines = [f"# Ecliptic-strip TNO search", "",
             f"- Fields: {len(fields)} along the ecliptic, {a.size} deg each, {a.spacing} deg spacing",
             f"- Centres: {fields}",
             f"- Total runtime: {(time.time()-t0)/60:.0f} min", ""]
    if combined.empty:
        lines.append("No candidates produced.")
    else:
        kcol = "known_object" if "known_object" in combined.columns else None
        ga = combined[combined["grade"] == "A"]
        new = ga[ga[kcol] == ""] if kcol else ga
        known = ga[ga[kcol] != ""] if kcol else ga.iloc[0:0]
        cols = ["field", "dist_au", "n_nights", "arc_days", "ra_ref", "dec_ref",
                "true_rate", "fwhm_min", "fwhm_max", "rms", "mag_mean", "best_filter"]
        cols = [c for c in cols if c in combined.columns]
        lines.append(f"## Grade A, UNMATCHED by SkyBoT: {len(new)} candidate(s) -- the shortlist")
        lines.append(new[cols].round(3).to_string(index=False) if len(new) else "(none)")
        lines.append("")
        lines.append(f"## Grade A, already-known objects: {len(known)}")
        if len(known):
            kc = cols + [kcol]
            lines.append(known[kc].round(3).to_string(index=False))
        lines.append("")
        nb_new = int((combined["grade"] == "B") & (combined.get(kcol, "") == "")) if kcol else len(combined[combined.grade=="B"])
        nb = combined[combined["grade"] == "B"]
        nb_new = len(nb[nb[kcol] == ""]) if kcol else len(nb)
        lines.append(f"## Grade B: {len(nb)} ({nb_new} unmatched) -- weaker, see strip_candidates.parquet")
        lines.append("")
    (out / "strip_report.md").write_text("\n".join(lines))
    log.info("STRIP COMPLETE in %.0f min -> %s", (time.time() - t0) / 60, out / "strip_report.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
