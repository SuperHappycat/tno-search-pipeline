"""Stage 4 - known-object cross-match and image-cutout verification of candidates.

For each candidate from Stage 3 this stage answers two questions:

1. Is it an already-catalogued minor planet? SkyBoT (IMCCE) lists every known solar-system
   object in a field at a given epoch; if a known object sits within `match_radius` of the
   candidate on two or more nights with the same designation, the candidate IS that object.
   (This is the expected outcome for most bright, slow movers - and recovering known objects
   is itself proof the pipeline works. A genuinely new object matches nothing.)

2. For the unmatched candidates, what do the pixels show? The NOIRLab SIA service returns
   image cutouts; a real object shows a point source that moves between nights, while cosmic
   rays, hot pixels and blends do not. The cutouts are written as a side-by-side PNG per
   candidate for visual inspection.

Network: SkyBoT and the SIA service are external, so run this on the machine that reached
NOIRLab for the download (the same venv as Stages 1-3), not in an offline sandbox.

Usage
-----
    python -m tno_pipeline.stage4_verify --ra 2.0 --dec 0.0                 # all grade-A/B candidates
    python -m tno_pipeline.stage4_verify --ra 2.0 --dec 0.0 --grades A      # only grade A
    python -m tno_pipeline.stage4_verify --ra 2.0 --dec 0.0 --no-cutouts    # cross-match only
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from .common import Config

log = logging.getLogger("stage4")


# ----------------------------------------------------------------------------
# 1. Known-object cross-match (SkyBoT cone search per night)
# ----------------------------------------------------------------------------
def crossmatch_skybot(nights, match_radius=10.0, observer="W84"):
    """For a list of (mjd, ra, dec) night positions, query SkyBoT and return the matched
    object name if the same designation lies within `match_radius` arcsec on >= 2 nights.

    observer defaults to W84 = CTIO (DECam). Returns (name or None, per-night details).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    try:
        from astroquery.imcce import Skybot
    except Exception as exc:                               # pragma: no cover
        raise RuntimeError("astroquery.imcce.Skybot unavailable; pip install -U astroquery") from exc

    import warnings
    hits = {}
    details = []
    for mjd, ra, dec in nights:
        c = SkyCoord(ra * u.deg, dec * u.deg)
        tbl = None
        try:                                               # SkyBoT: no object -> empty table or RuntimeError
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tbl = Skybot.cone_search(c, 120 * u.arcsec, Time(mjd, format="mjd"), location=observer)
        except Exception:
            tbl = None
        if tbl is None or len(tbl) == 0 or "RA" not in tbl.colnames or "DEC" not in tbl.colnames:
            details.append((mjd, None, None)); continue    # no known object near this night
        sep = c.separation(SkyCoord(tbl["RA"], tbl["DEC"]))
        k = int(np.argmin(sep))
        if sep[k] <= match_radius * u.arcsec:
            name = str(tbl["Name"][k])
            hits[name] = hits.get(name, 0) + 1
            details.append((mjd, name, float(sep[k].to("arcsec").value)))
        else:
            details.append((mjd, None, float(sep.min().to("arcsec").value)))
    matched = next((n for n, k in hits.items() if k >= 2), None)
    return matched, details


def annotate_known(cand: pd.DataFrame, orphans: pd.DataFrame, grades=("A", "B"),
                   match_radius=10.0, observer="W84") -> pd.DataFrame:
    """Add a `known_object` column: for each candidate in the given grades, cross-match its
    per-night observed positions against SkyBoT. Empty string means no known object (a real
    shortlist item); a name means it is that catalogued minor planet. Other grades get "".
    """
    o = orphans.set_index("id") if "id" in orphans.columns else orphans
    names = []
    for _, c in cand.iterrows():
        if c.get("grade") not in grades:
            names.append(""); continue
        sub = o.loc[[i for i in c["det_ids"] if i in o.index]].copy()
        sub["ngt"] = np.floor(sub["mjd"] - 0.7).astype(int)
        per = sub.groupby("ngt").agg(mjd=("mjd", "mean"), ra=("ra", "mean"), dec=("dec", "mean"))
        name, _ = crossmatch_skybot(list(per.itertuples(index=False, name=None)),
                                    match_radius, observer)
        names.append(name or "")
    cand = cand.copy(); cand["known_object"] = names
    return cand


# ----------------------------------------------------------------------------
# 2. Image cutouts (NOIRLab SIA) + PNG grid
# ----------------------------------------------------------------------------
def _sia_search(sia_url, ra, dec, size_deg, retries=3):
    """SIA cone search with retry/backoff on transient 5xx server errors."""
    import time
    import pyvo
    svc = pyvo.dal.SIAService(sia_url)
    for attempt in range(1, retries + 1):
        try:
            return svc.search(pos=(ra, dec), size=size_deg).to_table()
        except Exception as exc:
            msg = str(exc)
            if any(code in msg for code in ("502", "503", "504", "Bad Gateway")) and attempt < retries:
                time.sleep(5 * attempt); continue
            raise
    return None


SIA_COLLECTIONS = ["nsc_dr2", "des_dr2", "calibrated_all", "nsa"]


def sia_cutout(ra, dec, mjd, filt=None, size_pix=64, pixscale=0.263, day_tol=2.0,
               sia_url=None):
    """Try several SIA collections until one yields a same-night science cutout."""
    collections = [sia_url] if sia_url else SIA_COLLECTIONS
    for coll in collections:
        base = coll if str(coll).startswith("http") else f"https://datalab.noirlab.edu/sia/{coll}"
        img = _sia_cutout_one(ra, dec, mjd, filt, size_pix, pixscale, day_tol, base)
        if img is not None:
            return img
    return None


def _sia_cutout_one(ra, dec, mjd, filt, size_pix, pixscale, day_tol, sia_url):
    """Fetch the DECam science cutout from the SAME night as the detection.

    The NOIRLab `nsa` SIA v1 service returns every overlapping product across all epochs,
    including weight maps and data-quality masks. We keep only the calibrated single-epoch
    science image (prodtype='image', proctype='instcal') in the detection's band whose
    `mjd_obs` is within `day_tol` of the detection, then cut a stamp around (ra, dec).
    Returns a 2D array, or None if there is no same-night science image.
    """
    import numpy as np
    import pyvo
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.utils.data import download_file
    from astropy.wcs import WCS

    size_deg = size_pix * pixscale / 3600.0
    tbl = None
    for sz in (0.05, 0.15):                                # search box to find the overlapping exposure
        try:
            tbl = _sia_search(sia_url, ra, dec, sz)
        except Exception as exc:
            log.warning("SIA query failed at %.4f %.4f: %s", ra, dec, exc); return None
        if tbl is not None and len(tbl):
            break
    if tbl is None or len(tbl) == 0:
        return None

    cols = {c.lower(): c for c in tbl.colnames}
    def col(name):
        return np.asarray(tbl[cols[name]]) if name in cols else None
    m = np.ones(len(tbl), bool)
    pt = col("prodtype")
    if pt is not None:
        m &= np.char.lower(pt.astype(str)) == "image"      # drop wtmap / dqmask
    pc = col("proctype")
    if pc is not None:
        m &= np.isin(np.char.lower(pc.astype(str)), ["instcal"])   # calibrated single-epoch
    bp = col("obs_bandpass")
    if bp is not None and filt:
        m &= np.char.startswith(np.char.lower(bp.astype(str)), filt.lower())
    mj = col("mjd_obs")
    if mj is not None:
        mj = np.array([float(v) if str(v) not in ("", "--", "None", "nan") else np.nan for v in mj])
        m &= np.isfinite(mj) & (np.abs(mj - mjd) <= day_tol)   # same night
    idx = np.flatnonzero(m)
    nkept = len(idx)
    if nkept == 0:
        log.info("  no same-night science image at %.4f %.4f (%d products, none match band/epoch)",
                 ra, dec, len(tbl))
        return None
    if mj is not None:                                     # nearest in time among the matches
        idx = idx[np.argsort(np.abs(mj[idx] - mjd))]

    url_col = cols.get("access_url") or cols.get("accref") or cols.get("acref")
    for i in idx:
        try:
            url = str(tbl[url_col][int(i)])
            fpath = download_file(url, cache=True, show_progress=False, timeout=180)
            with fits.open(fpath) as hd:
                hdu = next(h for h in hd if getattr(h, "data", None) is not None and h.data.ndim == 2)
                cut = Cutout2D(hdu.data, SkyCoord(ra, dec, unit="deg"), size_pix,
                               wcs=WCS(hdu.header), mode="partial")
                data = np.asarray(cut.data, float)
            if np.isfinite(data).sum() > 0.2 * data.size:  # must actually cover the position
                return data
        except Exception as exc:
            log.debug("cutout attempt failed: %s", exc); continue
    return None


def cutout_grid(cand_row, orphans, outpath, size_pix=64, sia_url="https://datalab.noirlab.edu/sia/nsc_dr2"):
    """Make a one-row PNG of per-night cutouts centred on the candidate's track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = orphans.loc[cand_row["det_ids"]].copy()
    sub["ngt"] = np.floor(sub["mjd"] - 0.7).astype(int)
    per = sub.groupby("ngt").agg(mjd=("mjd", "mean"), ra=("ra", "mean"), dec=("dec", "mean"),
                                 filt=("filter", lambda s: s.mode().iloc[0]))
    n = len(per)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), squeeze=False)
    for ax, (ngt, r) in zip(axes[0], per.iterrows()):
        img = sia_cutout(r.ra, r.dec, r.mjd, filt=r.filt, size_pix=size_pix, sia_url=sia_url)
        if img is None:
            ax.text(0.5, 0.5, "no image", ha="center", va="center"); ax.set_facecolor("0.9")
        else:
            v = np.nanpercentile(img, [5, 99])
            ax.imshow(img, origin="lower", cmap="gray_r", vmin=v[0], vmax=v[1])
            ax.plot(size_pix / 2, size_pix / 2, "r+", ms=14, mew=1.5)      # predicted position
        ax.set_title(f"MJD {r.mjd:.1f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{outpath.stem}  dist~{cand_row['dist_au']:.0f}AU  {cand_row.get('best_filter','')}"
                 f"  mag~{cand_row['mag_mean']:.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120); plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
def diagnose_sia(cfg, sia_url="https://datalab.noirlab.edu/sia/nsa", grade="A"):
    """Print, for the first matching candidate's nights, what the SIA service returns:
    column names, how many images overlap, and each image's time vs the detection MJD.
    Use this to find the right time column and confirm same-epoch coverage.
    """
    import pyvo
    cand = pd.read_parquet(cfg.path("candidates.parquet"))
    cand = cand[cand.get("grade", "A") == grade].reset_index(drop=True)
    o = pd.read_parquet(cfg.path("orphans.parquet")).set_index("id")
    c = cand.iloc[0]
    sub = o.loc[c["det_ids"]].copy(); sub["ngt"] = np.floor(sub["mjd"] - 0.7).astype(int)
    per = sub.groupby("ngt").agg(mjd=("mjd", "mean"), ra=("ra", "mean"), dec=("dec", "mean"))
    svc = pyvo.dal.SIAService(sia_url)
    print(f"# SIA diagnostic for grade-{grade} candidate 0 at {sia_url}")
    for ngt, r in per.iterrows():
        res = svc.search(pos=(r.ra, r.dec), size=0.02)
        tbl = res.to_table()
        print(f"\n## night {ngt}  det MJD={r.mjd:.3f}  pos=({r.ra:.4f},{r.dec:.4f})  -> {len(tbl)} images")
        if len(tbl) == 0:
            continue
        print("   columns:", list(tbl.colnames))
        # show any time-like column values for the first few rows
        tcands = [col for col in tbl.colnames if any(k in col.lower()
                  for k in ("mjd", "date", "t_min", "obs", "dtpropid", "expnum"))]
        print("   time-like columns:", tcands)
        for col in tcands[:4]:
            vals = list(tbl[col][:6])
            print(f"     {col}: {vals}")
    return


def run(cfg: Config, grades=("A", "B"), do_cutouts=True, match_radius=10.0, observer="W84",
        sia_url="https://datalab.noirlab.edu/sia/nsc_dr2"):
    cfile = cfg.path("candidates.parquet")
    cand = pd.read_parquet(cfile)
    if "grade" in cand:
        cand = cand[cand["grade"].isin(list(grades))].reset_index(drop=True)
    orphans = pd.read_parquet(cfg.path("orphans.parquet")).set_index("id")
    log.info("verifying %d candidate(s) of grade %s", len(cand), "/".join(grades))

    outdir = cfg.path("cutouts"); outdir.mkdir(exist_ok=True)
    rows = []
    for i, c in cand.iterrows():
        sub = orphans.loc[c["det_ids"]].copy()
        sub["ngt"] = np.floor(sub["mjd"] - 0.7).astype(int)
        per = sub.groupby("ngt").agg(mjd=("mjd", "mean"), ra=("ra", "mean"), dec=("dec", "mean"))
        nights = list(per.itertuples(index=False, name=None))
        name, details = crossmatch_skybot(nights, match_radius, observer)
        status = f"KNOWN: {name}" if name else "unmatched"
        log.info("cand %d  dist~%.0fAU  grade %s -> %s", i, c["dist_au"], c.get("grade", "?"), status)
        png = ""
        if name is None and do_cutouts:
            try:
                png = str(cutout_grid(c, orphans, outdir / f"cand_{i:04d}.png", sia_url=sia_url))
            except Exception as exc:
                log.warning("cutout grid failed for cand %d: %s", i, exc)
        rows.append(dict(index=int(i), grade=c.get("grade", "?"), dist_au=c["dist_au"],
                         ra_ref=c["ra_ref"], dec_ref=c["dec_ref"], mag_mean=c["mag_mean"],
                         known_object=name or "", status=status, cutout_png=png))
    rep = pd.DataFrame(rows)
    rep.to_parquet(cfg.path("verification.parquet"), index=False)
    log.info("wrote %s (%d unmatched -> inspect cutouts in %s)",
             cfg.path("verification.parquet"), int((rep["known_object"] == "").sum()), outdir)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, default=Config.ra0)
    p.add_argument("--dec", type=float, default=Config.dec0)
    p.add_argument("--outdir", default="tno_data")
    p.add_argument("--grades", default="A,B", help="candidate grades to verify, e.g. A or A,B")
    p.add_argument("--match-radius", type=float, default=10.0, help="SkyBoT match radius [arcsec]")
    p.add_argument("--observer", default="W84", help="IAU observatory code (W84 = CTIO/DECam)")
    p.add_argument("--no-cutouts", action="store_true", help="cross-match only, skip image download")
    p.add_argument("--sia-url", default="https://datalab.noirlab.edu/sia/nsc_dr2",
                   help="NOIRLab SIA v1 collection endpoint (nsc_dr2 holds the NSC exposures; also nsa, des_dr2)")
    p.add_argument("--diag", action="store_true",
                   help="print what the SIA service returns for one candidate's nights, then exit")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config(ra0=a.ra, dec0=a.dec, outdir=a.outdir)
    if a.diag:
        diagnose_sia(cfg, sia_url=a.sia_url, grade=a.grades.split(",")[0])
        return
    rep = run(cfg, grades=tuple(a.grades.split(",")), do_cutouts=not a.no_cutouts,
              match_radius=a.match_radius, observer=a.observer, sia_url=a.sia_url)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(rep.drop(columns="cutout_png").to_string())


if __name__ == "__main__":
    main()
