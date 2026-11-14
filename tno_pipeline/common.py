"""Shared configuration and geometry helpers for the TNO pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARCSEC_PER_RAD = 206264.80624709636


@dataclass
class Config:
    # ---- field ---------------------------------------------------------
    ra0: float = 2.0            # field centre RA  [deg, ICRS]
    dec0: float = 0.0           # field centre Dec [deg, ICRS]
    size_deg: float = 1.0       # on-sky side length of the square patch [deg]
    n_tiles: int = 4            # patch is queried as n_tiles x n_tiles sub-boxes

    # ---- TAP -----------------------------------------------------------
    tap_url: str = "https://datalab.noirlab.edu/tap"
    table: str = "nsc_dr2.meas"
    spatial_mode: str = "box"   # "box" (pure ADQL) | "q3c" | "cone"
    decam_only: bool = True     # keep exposures whose name starts with "c4d"
    max_retries: int = 3

    # ---- detection quality cuts (applied locally, after download) ------
    max_magerr: float = 0.35    # ~3 sigma; faint TNOs live near the limit
    min_fwhm: float = 0.5       # arcsec; rejects cosmic rays / hot pixels
    max_fwhm: float = 3.0       # arcsec; rejects blends and extended junk
    require_flags_zero: bool = False

    # ---- Stage 1: stationary-source rejection --------------------------
    stationary_radius: float = 1.5   # arcsec
    night_offset: float = 0.7        # night = floor(mjd - offset); 0.7 d puts
                                     # the boundary at local noon on Cerro Tololo

    # ---- Stage 2: tracklet linking -------------------------------------
    min_rate: float = 0.5       # arcsec / hour
    max_rate: float = 5.0       # arcsec / hour
    min_dt_hr: float = 0.25     # minimum time between two linked detections
    min_sep: float = 0.4        # arcsec; minimum displacement (astrometric noise floor)
    link_tol: float = 0.35      # arcsec; max residual from a straight line
    min_det: int = 3            # 3 = triplets only; 2 also emits bare pairs
    max_dmag_same: float = 0.7  # |dmag| allowed within the same filter
    max_dmag_diff: float = 1.5  # |dmag| allowed across different filters
    max_ecl_angle: float | None = None  # deg; if set, motion must lie within this
                                        # angle of the ecliptic direction
    workers: int = 1

    # ---- paths ---------------------------------------------------------
    outdir: Path = field(default_factory=lambda: Path("tno_data"))

    @property
    def tag(self) -> str:
        return f"ra{self.ra0:08.4f}_dec{self.dec0:+08.4f}"

    def path(self, name: str) -> Path:
        d = Path(self.outdir) / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d / name


def tangent_plane(ra, dec, ra0, dec0):
    """Gnomonic projection. Returns (xi, eta) in arcsec; xi is east, eta north."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cosc
    return xi * ARCSEC_PER_RAD, eta * ARCSEC_PER_RAD


def inverse_tangent_plane(xi, eta, ra0, dec0):
    """Inverse gnomonic projection. (xi, eta) in arcsec -> (ra, dec) in deg."""
    xi, eta = np.asarray(xi) / ARCSEC_PER_RAD, np.asarray(eta) / ARCSEC_PER_RAD
    ra0r, dec0r = np.radians(ra0), np.radians(dec0)
    denom = np.cos(dec0r) - eta * np.sin(dec0r)
    ra = ra0r + np.arctan2(xi, denom)
    dec = np.arctan2(np.sin(dec0r) + eta * np.cos(dec0r), np.hypot(xi, denom))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def night_id(mjd, offset: float = 0.7):
    """Integer observing-night label. See Config.night_offset."""
    return np.floor(np.asarray(mjd, dtype=float) - offset).astype(np.int64)


def ecliptic_position_angle(ra0: float, dec0: float) -> float:
    """Position angle (deg, east of north) of increasing ecliptic longitude at the field centre."""
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

    c = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
    e = c.transform_to(BarycentricMeanEcliptic())
    c2 = SkyCoord(e.lon + 0.01 * u.deg, e.lat, frame=BarycentricMeanEcliptic()).icrs
    return float(c.position_angle(c2).deg)


def ecliptic_latitude(ra0: float, dec0: float) -> float:
    import astropy.units as u
    from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord

    return float(SkyCoord(ra0 * u.deg, dec0 * u.deg).transform_to(BarycentricMeanEcliptic()).lat.deg)


# ---------------------------------------------------------------------------
# Stage 3 geometry: observer position and the heliocentric (de-parallax) transform
# ---------------------------------------------------------------------------
def _equatorial_basis(ra0, dec0):
    """Line-of-sight, east and north unit vectors (equatorial ICRS cartesian) at a point."""
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    n = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    e = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])
    return n, e, north


def observer_tangential(mjd, ra0, dec0):
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    from astropy.time import Time

    with solar_system_ephemeris.set("builtin"):
        t = Time(np.asarray(mjd, dtype=float), format="mjd")
        helio = (get_body_barycentric("earth", t) - get_body_barycentric("sun", t))
        X = helio.xyz.to(u.AU).value                      # (3, N) ICRS cartesian, AU
    _, e, north = _equatorial_basis(ra0, dec0)
    return e @ X, north @ X                               # each (N,)


def circular_sky_rate(dist_au: float) -> float:
    """Transverse sky rate (arcsec/day) of a circular heliocentric orbit at `dist_au`.

    Used as the scale for the bound-orbit velocity prior in Stage 3. The true motion
    after de-parallaxing is at most a small multiple of this.
    """
    vcirc_km_s = 29.78 / np.sqrt(dist_au)
    return vcirc_km_s / (dist_au * 1.495978707e8) * ARCSEC_PER_RAD * 86400.0
