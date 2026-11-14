# TNO Search Pipeline

Local Python pipeline to discover faint, slow-moving Trans-Neptunian Objects in
archival DECam data from the NOIRLab Source Catalog (NSC DR2).

Four stages:
1. `stage1_ingest` — TAP query of NSC DR2 + stationary-source rejection.
2. `stage2_tracklets` — intra-night tracklet linking.
3. `stage3_linking` — HelioLinC-style multi-night orbit linking (de-parallax + clustering),
   with brightness-, bound-orbit- and PSF-consistency grading.
4. `stage4_verify` — SkyBoT known-object cross-match + DECam image cutouts.

`strip_search` orchestrates a multi-field ecliptic-strip search end to end.

## Setup
    python3 -m venv tno && source tno/bin/activate
    pip install -r requirements.txt

## Run a strip (cloud or local)
    python3 -m tno_pipeline.strip_search --ra0 2.0 --dec0 0.0 --nfields 6 --delete-tiles

Needs network access to datalab.noirlab.edu (TAP + SIA) and vo.imcce.fr (SkyBoT).

## Offline checks
    python3 -m tno_pipeline.selftest
    python3 -m tno_pipeline.selftest3
