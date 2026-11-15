"""Deep-field TNO search: one heavily-imaged ecliptic field, where a real object is
detected on many nights and rises far above the 3-night chance-alignment floor.

Ranks candidates by number of nights (the real discriminator), cross-matches the top
tracks against known minor planets per-detection (a line-up of different known asteroids
is rejected, not just repeats of one), and makes cutouts for the standout unmatched tracks.

    python3 -m tno_pipeline.deep_search --ra 0.0 --dec 0.0 --size 0.6
"""
from __future__ import annotations
import argparse, logging, numpy as np, pandas as pd
from .common import Config
from . import stage1_ingest, stage3_linking, stage4_verify
log = logging.getLogger("deep")


def run(ra, dec, size, outdir, max_window, min_nights_report, topN):
    cfg = Config(ra0=ra, dec0=dec, size_deg=size, n_tiles=4, outdir=outdir)
    orph = stage1_ingest.run(cfg)
    cand = stage3_linking.run(cfg, [str(cfg.path("orphans.parquet"))],
        dmin=35.0, dmax=120.0, dstep=2.0, window=None, step=None, window_scale=1.5,
        max_window=max_window, vprior=2.0, pos_cell=3.0, vel_cells=8.0, min_det=3,
        min_nights=3, rms_tol=0.6, collapse=True, collapse_radius=1.5, max_dmag=1.0,
        bound_factor=1.5)
    c = pd.read_parquet(cfg.path("candidates.parquet"))
    log.info("deep field: %d candidates; n_nights max=%d, distribution=%s",
             len(c), int(c["n_nights"].max()) if len(c) else 0,
             dict(sorted(c["n_nights"].value_counts().items())) if len(c) else {})

    # the real signal: tracks on many nights. Rank by n_nights, then arc, then rms.
    top = c.sort_values(["n_nights", "arc_days", "rms"], ascending=[False, False, True]).head(topN).reset_index(drop=True)
    # per-detection known-object cross-match (corrected logic) on the top tracks
    o = orph.set_index("id")
    import astropy.units as u, warnings; warnings.simplefilter("ignore")
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astroquery.imcce import Skybot
    tags = []
    for _, row in top.iterrows():
        sub = o.loc[[i for i in row["det_ids"] if i in o.index]].copy()
        sub["ngt"] = np.floor(sub["mjd"] - 0.7).astype(int)
        per = sub.groupby("ngt").agg(mjd=("mjd", "mean"), ra=("ra", "mean"), dec=("dec", "mean"))
        nmatch = 0
        for _, r in per.iterrows():
            sc = SkyCoord(r.ra * u.deg, r.dec * u.deg)
            try:
                t = Skybot.cone_search(sc, 60 * u.arcsec, Time(r.mjd, format="mjd"), location="W84")
                if sc.separation(SkyCoord(t["RA"], t["DEC"])).min().to("arcsec").value <= 8:
                    nmatch += 1
            except Exception:
                pass
        frac = nmatch / len(per)
        tags.append("known-asteroid line-up" if frac >= 0.7 else
                    ("mixed/partly-known" if frac >= 0.34 else "UNMATCHED"))
    top["known_frac_tag"] = tags

    # cutouts for the standout UNMATCHED many-night tracks
    outc = cfg.path("cutouts"); outc.mkdir(exist_ok=True)
    shortlist = top[(top["known_frac_tag"] == "UNMATCHED") & (top["n_nights"] >= min_nights_report)]
    for i, row in shortlist.iterrows():
        try:
            stage4_verify.cutout_grid(row, o, outc / f"deep_cand_{i:04d}.png")
        except Exception as e:
            log.warning("cutout failed cand %d: %s", i, e)

    cols = ["n_nights", "n_obs", "arc_days", "dist_au", "ra_ref", "dec_ref", "true_rate",
            "fwhm_min", "fwhm_max", "rms", "mag_mean", "best_filter", "grade", "known_frac_tag"]
    cols = [x for x in cols if x in top.columns]
    rep = [f"# Deep-field TNO search  (RA {ra}, Dec {dec}, {size} deg)", "",
           f"- {len(c)} candidates; max n_nights = {int(c['n_nights'].max()) if len(c) else 0}",
           f"- A real object shows as a many-night track; chance alignments stay at 3 nights.", "",
           f"## Top {len(top)} tracks by number of nights", "",
           top[cols].round(3).to_string(index=False), "",
           f"## Standout UNMATCHED tracks (n_nights >= {min_nights_report}): {len(shortlist)}",
           (shortlist[cols].round(3).to_string(index=False) if len(shortlist) else "(none — nothing rose above the floor unmatched)")]
    (cfg.path("deep_report.md")).write_text("\n".join(rep))
    top.to_parquet(cfg.path("deep_top.parquet"), index=False)
    print("\n".join(rep))
    log.info("DEEP DONE -> %s", cfg.path("deep_report.md"))
    return top


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ra", type=float, default=0.0); p.add_argument("--dec", type=float, default=0.0)
    p.add_argument("--size", type=float, default=0.6); p.add_argument("--outdir", default="tno_deep")
    p.add_argument("--max-window", type=float, default=55.0)
    p.add_argument("--min-nights-report", type=int, default=5)
    p.add_argument("--topN", type=int, default=25)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(a.ra, a.dec, a.size, a.outdir, a.max_window, a.min_nights_report, a.topN)


if __name__ == "__main__":
    main()
