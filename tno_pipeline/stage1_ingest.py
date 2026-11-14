"""Stage 1 - TAP ingestion of NSC DR2 measurements and stationary-source rejection.

Usage
-----
    python -m tno_pipeline.stage1_ingest --ra 2.0 --dec 0.0 --size 1.0
    python -m tno_pipeline.stage1_ingest --describe      # print the table's columns
    python -m tno_pipeline.stage1_ingest --probe         # test which query form the server accepts

Outputs (under <outdir>/<field tag>/):
    tiles/tile_XX_YY.parquet   raw per-tile downloads (cache; delete to re-query)
    detections.parquet         all detections passing quality cuts, with `stationary` flag
    orphans.parquet            single-night detections only (input to Stage 2)
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .common import Config, ecliptic_latitude, night_id, tangent_plane

log = logging.getLogger("stage1")

# NSC DR2 `meas` has no column called `id`; the per-measurement key is `measid`
# (aliased to `id` on output) and `objectid` is NSC's own cross-epoch association.
COLUMNS = [
    "measid", "objectid", "exposure", "ccdnum", "filter", "mjd",
    "ra", "dec", "raerr", "decerr",
    "mag_auto", "magerr_auto", "fwhm", "class_star", "flags",
]


# ----------------------------------------------------------------------------
# Query construction
# ----------------------------------------------------------------------------
def make_tiles(cfg: Config):
    """Split the patch into n x n (ra_min, ra_max, dec_min, dec_max) boxes.

    The RA half-width is divided by cos(dec) so the patch is `size_deg` wide on the sky.
    RA limits are returned un-wrapped (may be <0 or >360); build_adql handles the wrap.
    """
    half = cfg.size_deg / 2.0
    half_ra = half / np.cos(np.radians(cfg.dec0))
    ra_edges = np.linspace(cfg.ra0 - half_ra, cfg.ra0 + half_ra, cfg.n_tiles + 1)
    dec_edges = np.linspace(cfg.dec0 - half, cfg.dec0 + half, cfg.n_tiles + 1)
    return [
        (i, j, ra_edges[i], ra_edges[i + 1], dec_edges[j], dec_edges[j + 1])
        for i in range(cfg.n_tiles) for j in range(cfg.n_tiles)
    ]


def build_adql(cfg: Config, ra_min, ra_max, dec_min, dec_max, quote: bool = False, top: int | None = None) -> str:
    """ADQL for one box. `quote=True` writes "dec" and "filter" as delimited identifiers,
    for TAP parsers that treat those two words as reserved."""
    q = (lambda c: f'"{c}"' if c in ("dec", "filter") else c) if quote else (lambda c: c)
    dec = q("dec")
    cols = ", ".join(q(c) for c in COLUMNS)
    a, b = ra_min % 360.0, ra_max % 360.0
    ra_box = (f"ra >= {a:.8f} AND ra < {b:.8f}" if a < b
              else f"(ra >= {a:.8f} OR ra < {b:.8f})")          # tile straddles RA = 0
    box = f"{ra_box} AND {dec} >= {dec_min:.8f} AND {dec} < {dec_max:.8f}"

    if cfg.spatial_mode == "box":
        where = box
    else:
        # Circumscribing cone (uses the server's spatial index), trimmed to the box.
        rac, decc = (ra_min + ra_max) / 2.0 % 360.0, (dec_min + dec_max) / 2.0
        rad = 1.01 * 0.5 * np.hypot((ra_max - ra_min) * np.cos(np.radians(decc)), dec_max - dec_min)
        if cfg.spatial_mode == "q3c":
            cone = f"'t' = q3c_radial_query(ra, {dec}, {rac:.8f}, {decc:.8f}, {rad:.8f})"
        elif cfg.spatial_mode == "cone":
            cone = (f"1 = CONTAINS(POINT('ICRS', ra, {dec}), "
                    f"CIRCLE('ICRS', {rac:.8f}, {decc:.8f}, {rad:.8f}))")
        else:
            raise ValueError(f"unknown spatial_mode {cfg.spatial_mode!r}")
        where = f"{cone} AND {box}"
    head = f"SELECT TOP {top} " if top else "SELECT "
    return f"{head}{cols} FROM {cfg.table} WHERE {where}"


# ----------------------------------------------------------------------------
# TAP access
# ----------------------------------------------------------------------------
# (transport, quote identifiers). Tried in this order; the first that works is used.
VARIANTS = [("sync", False), ("async", False), ("sync", True), ("async", True)]


def _service(cfg: Config):
    import pyvo
    return pyvo.dal.TAPService(cfg.tap_url)


def _error_text(exc: Exception) -> str:
    """Server's own explanation where there is one (pyvo keeps it; bare HTTP reasons do not)."""
    resp = getattr(exc, "response", None)
    body = getattr(resp, "text", "") if resp is not None else ""
    return " | ".join(x for x in (str(exc).strip(), body.strip()[:600]) if x)


def tap_query(cfg: Config, adql: str, transport: str = "sync"):
    """Run one query. Returns (DataFrame, overflowed)."""
    svc = _service(cfg)
    res = svc.run_sync(adql) if transport == "sync" else svc.run_async(adql)
    try:
        overflow = str(getattr(res, "query_status", None) or res.status[0]).upper() == "OVERFLOW"
    except Exception:
        overflow = False
    df = res.to_table().to_pandas()
    for c in df.columns:                          # VOTable strings may arrive as bytes
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df, overflow


def probe(cfg: Config, stop_at_first: bool = True):
    """Send a tiny TOP 1 query in each variant and report which the server accepts.

    Returns the first working (transport, quote). With stop_at_first=False every variant
    is tried and logged, which is the diagnostic to send back if downloads fail.
    """
    d = 0.02
    working, failures = None, []
    for transport, quote in VARIANTS:
        adql = build_adql(cfg, cfg.ra0 - d, cfg.ra0 + d, cfg.dec0 - d, cfg.dec0 + d, quote=quote, top=1)
        label = f"{transport}{' + quoted identifiers' if quote else ''}"
        try:
            t0 = time.time()
            df, _ = tap_query(cfg, adql, transport)
            log.info("probe %-28s OK   (%d row, %.1f s)", label, len(df), time.time() - t0)
            working = working or (transport, quote)
            if stop_at_first:
                return working
        except Exception as exc:
            msg = _error_text(exc)
            failures.append(f"{label}: {msg}")
            log.warning("probe %-28s FAIL %s", label, msg)
    if working is None:
        raise SystemExit("The TAP service rejected every query variant:\n  " + "\n  ".join(failures)
                         + f"\nExample query:\n  {adql}")
    return working


def fetch_box(cfg: Config, box, variant, depth: int = 0) -> pd.DataFrame:
    """Download one box; if the server truncates the result (row limit), split it in four."""
    transport, quote = variant
    adql = build_adql(cfg, *box, quote=quote)
    last = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            df, overflow = tap_query(cfg, adql, transport)
            break
        except Exception as exc:
            last = exc
            log.warning("TAP attempt %d/%d failed: %s", attempt, cfg.max_retries, _error_text(exc))
            time.sleep(10 * attempt)
    else:
        raise RuntimeError(f"TAP query failed after {cfg.max_retries} attempts:\n{adql}") from last
    if not overflow:
        return df
    if depth >= 4:
        raise RuntimeError(f"result still truncated after {depth} subdivisions:\n{adql}")
    log.info("result truncated at %d rows; splitting box", len(df))
    r0, r1, d0, d1 = box
    rm, dm = (r0 + r1) / 2.0, (d0 + d1) / 2.0
    parts = [fetch_box(cfg, b, variant, depth + 1)
             for b in ((r0, rm, d0, dm), (rm, r1, d0, dm), (r0, rm, dm, d1), (rm, r1, dm, d1))]
    return pd.concat(parts, ignore_index=True)


def describe_table(cfg: Config) -> pd.DataFrame:
    """Return the server-side column list, to confirm COLUMNS matches the live schema."""
    q = (f"SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
         f"WHERE table_name = '{cfg.table}'")
    return tap_query(cfg, q, "sync")[0]


def download_field(cfg: Config) -> pd.DataFrame:
    """Download every tile (cached on disk, so an interrupted run resumes) and concatenate."""
    tile_dir = cfg.path("tiles")
    tile_dir.mkdir(exist_ok=True)
    frames, variant = [], None
    for i, j, r0, r1, d0, d1 in make_tiles(cfg):
        f = tile_dir / f"tile_{i:02d}_{j:02d}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            variant = variant or probe(cfg)
            t0 = time.time()
            df = fetch_box(cfg, (r0, r1, d0, d1), variant)
            df.to_parquet(f, index=False)
            log.info("tile (%d,%d): %d rows in %.0f s", i, j, len(df), time.time() - t0)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("measid").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Local filtering
# ----------------------------------------------------------------------------
def quality_cuts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n0 = len(df)
    m = np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["mjd"])
    m &= df["magerr_auto"].between(0, cfg.max_magerr) & (df["mag_auto"] < 50)
    m &= df["fwhm"].between(cfg.min_fwhm, cfg.max_fwhm)
    if cfg.decam_only:
        m &= df["exposure"].astype(str).str.startswith("c4d")
    if cfg.require_flags_zero:
        m &= df["flags"] == 0
    log.info("quality cuts: %d -> %d detections", n0, int(m.sum()))
    return df[m].reset_index(drop=True)


def flag_stationary(x, y, night, radius: float) -> np.ndarray:
    """True where a detection has a neighbour within `radius` arcsec on a *different* night.

    For each detection we count neighbours over all nights and neighbours on its own
    night; any excess means another night saw a source at that position. This needs
    no pair list, so memory stays O(N) even for sources with hundreds of epochs.
    Same-night repeats are deliberately kept: a 1"/hr TNO re-observed after 30 min
    moves only 0.5" and must not be discarded here.
    """
    xy = np.column_stack([x, y])
    n_all = cKDTree(xy).query_ball_point(xy, radius, return_length=True, workers=-1)
    n_same = np.zeros(len(xy), dtype=np.int64)
    order = np.argsort(night, kind="stable")
    bounds = np.flatnonzero(np.diff(night[order])) + 1
    for idx in np.split(order, bounds):
        sub = xy[idx]
        n_same[idx] = cKDTree(sub).query_ball_point(sub, radius, return_length=True)
    return n_all > n_same


def find_orphans(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add projection / night columns and the `stationary` flag. Returns the full table."""
    df = df.copy()
    df["x"], df["y"] = tangent_plane(df["ra"].values, df["dec"].values, cfg.ra0, cfg.dec0)
    df["night"] = night_id(df["mjd"].values, cfg.night_offset)
    df["stationary"] = flag_stationary(df["x"].values, df["y"].values,
                                       df["night"].values, cfg.stationary_radius)
    log.info("stationary filter: %d of %d detections are single-night orphans (%d nights)",
             int((~df["stationary"]).sum()), len(df), df["night"].nunique())
    return df


def run(cfg: Config) -> pd.DataFrame:
    log.info("field %s, ecliptic latitude %.2f deg", cfg.tag, ecliptic_latitude(cfg.ra0, cfg.dec0))
    df = download_field(cfg)
    if df.empty:
        raise SystemExit("No NSC DR2 measurements in this patch - choose another field.")
    df = find_orphans(quality_cuts(df, cfg), cfg).rename(columns={"measid": "id"})
    df.to_parquet(cfg.path("detections.parquet"), index=False)
    orphans = df[~df["stationary"]].drop(columns="stationary").reset_index(drop=True)
    orphans.to_parquet(cfg.path("orphans.parquet"), index=False)
    per_night = orphans.groupby("night")["exposure"].nunique()
    log.info("wrote %s (%d orphans; %d nights have >=3 exposures)",
             cfg.path("orphans.parquet"), len(orphans), int((per_night >= 3).sum()))
    return orphans


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, default=Config.ra0)
    p.add_argument("--dec", type=float, default=Config.dec0)
    p.add_argument("--size", type=float, default=Config.size_deg, help="patch side [deg]")
    p.add_argument("--tiles", type=int, default=Config.n_tiles)
    p.add_argument("--spatial-mode", choices=["box", "q3c", "cone"], default=Config.spatial_mode)
    p.add_argument("--all-instruments", action="store_true", help="keep Mosaic3 / 90Prime too")
    p.add_argument("--outdir", default="tno_data")
    p.add_argument("--describe", action="store_true", help="print live table columns and exit")
    p.add_argument("--probe", action="store_true", help="test every query variant and exit")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = Config(ra0=a.ra, dec0=a.dec, size_deg=a.size, n_tiles=a.tiles,
                 spatial_mode=a.spatial_mode, decam_only=not a.all_instruments, outdir=a.outdir)
    if a.describe:
        print(describe_table(cfg).to_string())
        return
    if a.probe:
        print("first working variant:", probe(cfg, stop_at_first=False))
        return
    run(cfg)


if __name__ == "__main__":
    main()
