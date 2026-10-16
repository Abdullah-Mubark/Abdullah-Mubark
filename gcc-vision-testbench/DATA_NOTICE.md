# Data notice

This repository contains **code only**. Data collected with it stays on your machine under
`data/` (git-ignored) and is governed by the terms below. You are responsible for following them.

## Third-party datasets keep their original licences

- **ChokePoint** (Zenodo record 815657): use is subject to the licence and citation requirements
  on the Zenodo record page. Cite the original authors in any publication.
- **Roboflow Universe datasets**: each dataset has its own licence (often CC BY 4.0, but not
  always). The collector records the licence in `manifest.csv` when Roboflow reports it. Check
  the licence before any use beyond internal testing, and give attribution where required.

Downloading a dataset with these scripts does not change its licence.

## Model weights used for pre-annotation

Detector weights also come with their own licences. Ultralytics YOLOv8 code and its official
weights are AGPL-3.0. InsightFace's pretrained model packs are for non-commercial research only.
Draft labels produced with these models are still your own data, but check the licence before
using a model in a product.

## YouTube clips: internal testing only

Clips taken from YouTube are third-party copyrighted material, and downloading them may conflict
with YouTube's Terms of Service. In this project they are used **only for internal evaluation of
model accuracy**. They must **not** be redistributed, published, shared outside the team, uploaded
elsewhere, or used to train models you release. Delete them when they are no longer needed for
testing. Every clip's metadata JSON records its source URL so you can trace and remove it.

## Self-recorded footage needs consent

Video of identifiable people (faces) and of vehicle licence plates is **personal data** under the
**Saudi Personal Data Protection Law (PDPL)** and the **UAE Federal Decree-Law No. 45 of 2021 on
the Protection of Personal Data (PDPL)**; other GCC states have similar laws. Before recording or
importing footage:

- get informed consent from the people recorded (or rely on another lawful basis your legal team
  has confirmed), and state the purpose (testing recognition systems);
- don't record where people expect privacy, and follow local rules on filming in public places;
- keep the footage secured, access-limited, and only as long as needed;
- honour requests to delete someone's data, using the `sha256` and `original_path` fields in each
  clip's metadata to find it.

`local_import.py` will not run until you confirm this with `consent_ack: true` in
`config/local_sources.yaml` or `--consent-ack`.

This notice is not legal advice. Consult your organisation's data-protection officer.
