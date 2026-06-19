# fitCSG

Fit a **Constructive Solid Geometry (CSG) tree of signed-distance primitives**
to a 3D object, as a compact, interpretable, editable, grasp-aware shape
abstraction (the work behind the *CSGGrasp* submission).

Given a tree **topology** (which primitives + boolean ops, authored by hand or
by an LLM), the continuous **leaf parameters** (center / size / rotation / …)
are optimised so the tree's SDF matches a target SDF derived from an observed
object.

![CSG fitting a point cloud](assets/fit_demo.gif)

*Above: the data-free demo (`examples/mug.json`). A deliberately-randomised CSG
tree (colour) is optimised until its surface matches the target point cloud
(grey): a hollow cylinder body, a subtracted cavity, and a rotated torus
handle.*

> **Status.** This was research code that was buggy and partly faked-to-work; it
> was rebuilt in June 2026 into a clean, *correct* implementation of the idea
> (`fitcsg/` package + tests). It runs CPU-only; a GPU just speeds up fitting.
> The synthetic demo is fully reproducible; the real-data path is implemented
> but untested (no captured data is currently available — see TODOs).

## Pipeline

```
RGB image --(external)--> point cloud --> target SDF  ─┐
                                                       ├─> optimise CSG leaf params  (this repo)
hand-/LLM-authored CSG tree --> predicted SDF  ────────┘
```

This repo owns: **CSG → SDF**, **point cloud → target SDF**, and the
**optimisation loop**. The `RGB → point cloud` front-end and the LLM topology
proposer live outside the repo.

## Layout

```
fitcsg/
  transforms.py   Euler(deg) rotations + world->local transform (one convention everywhere)
  primitives.py   SDF primitives: sphere, ellipsoid, box, cylinder, cone, torus, capsule
  csg.py          tree parse/serialise, CSG ops (+ smooth blends), SDF evaluation, colours
  grid.py         dense grid sampling + surface extraction
  alignment.py    similarity ICP (point-cloud alignment)
  target.py       masked point cloud -> target SDF supervision
  synthetic.py    sample a target SDF from a known tree (no external data)
  optimize.py     fitting loop: truncated-Huber loss, cosine LR, random restarts
  visualize.py    Graphviz tree + matplotlib SDF/fit rendering + GIF assembly
  random_tree.py  random tree generation (smoke tests)
scripts/
  visualize_tree.py   render a tree's graph and/or SDF
  fit.py              fit a tree to a synthetic or real target
  fit_demo.py         animated fit -> GIF (the visual smoke test)
examples/
  sunglasses.json     original hand-authored example (converted to new schema)
  mug.json            demo: cylinder - cavity + torus handle (rotation, subtraction, torus)
tests/                pytest suite (run: pytest)
```

## Installation

```bash
conda env create -f environment.yml
conda activate fitcsg
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
```

## Quickstart

```bash
# Visualise a tree's SDF (no GPU)
python scripts/visualize_tree.py --tree examples/mug.json --save mug.png

# Self-contained fit (treat the tree as GT, randomise params, recover them)
python scripts/fit.py --tree examples/mug.json --num_steps 1500 --restarts 4

# Animated GIF of the fit (target cloud in grey, CSG surface in colour)
python scripts/fit_demo.py --tree examples/mug.json --num_steps 500 --outdir demo_out

# Fit to a real observation (needs your own data; see TODOs)
python scripts/fit.py --target files --tree examples/mug.json \
    --pc pointcloud.npy --mask mask.npy

# Tests
pytest
```

## Tests

`pytest` (15 tests) covers the correctness claims above so the next person can
refactor safely:

* `test_transforms.py` — rotation matrices are orthonormal with `det=1`;
  world→local preserves distances.
* `test_primitives.py` — exact sphere/box distances; every primitive encloses a
  non-empty solid and reports far points as outside; box rotation-equivariance;
  size-sign invariance.
* `test_csg.py` — union/intersection/subtraction signs; overlapping-sphere
  volumes; JSON round-trip; legacy-schema loading; colour output shape.
* `test_optimize.py` — the synthetic fit reduces loss by >2× and converges.
* `test_random_tree.py` — random trees parse and yield a non-empty solid.

## Conventions

* **SDF.** Negative = inside, positive = outside. Box/cylinder/sphere/torus/
  capsule are exact; cone is the exact capped-cone formula; the **ellipsoid is
  the standard Inigo-Quilez approximation** (no closed-form SDF exists — it is
  exactly 0 at the centre and accurate near the surface). CSG `min`/`max` give
  the correct sign and zero level set but are not exact distances away from the
  surface (inherent to CSG).
* **Pose.** Every leaf has `center` (world origin) and `rotation` (XYZ Euler
  angles in **degrees**), applied to *all* shapes via one world→local transform.
* **Positivity.** Sizes/radii are passed through `abs()` inside the SDF, so the
  optimiser is unconstrained and cannot invert a shape.
* **JSON schema.** `{"op", "left", "right", "smooth"?}` for internal nodes and
  `{"shape", "name", "params": {...}}` for leaves. Legacy keys
  (`operation`/`type`/`sizes`/`axis`) still load.

## What the rebuild fixed

Brief provenance (the original bugs): rotation was ignored for some primitives
and inconsistent for others; the JSON `rotation` key didn't match the `axis` key
the SDFs read (the shipped example would have crashed); `random_tree_utils.py`
couldn't import; leaf lookup broke for ≥10 instances of a shape; viz colours
were non-deterministic. All fixed, plus a robust loss, restarts, more primitives
(torus/capsule), and a test suite.

## Known limitations & flags

Things to be aware of before relying on this for the resubmission:

* **Coarse alignment is NOT implemented — it's a TODO.** Right now the object's
  canonical pose is *assumed*; the only alignment we do is a **local similarity
  ICP** (`alignment.py`) that lines up scale/rotation/translation between the
  CSG model surface and the observed cloud. It needs a good initial pose and
  diverges under large offsets (verified). A real coarse alignment / pose
  estimator is roadmap item #3 below.
* **Real-data path is untested.** `target.build_target_from_files` is wired up
  for `(pointcloud.npy, mask.npy)` but the original capture data is gone, so it
  has only been exercised on synthetic targets.
* **GPU path unexercised.** Developed/tested CPU-only; `--device cuda` should
  work but hasn't been run here.
* **Approximations.** The ellipsoid SDF and the CSG `min`/`max` are not exact
  distances away from the surface (see Conventions) — fine for fitting, but
  don't treat the field as a metric SDF everywhere.

## Large TODOs & extensions (research roadmap)

These are the substantial pieces left for the resubmission, roughly in order:

1. **Real-world experiments.** Re-establish the `RGB → point cloud` front-end and
   collect/curate objects with hand-authored trees + good initial params. The
   `point cloud → target SDF` half (`target.py`) is ready for
   `(pointcloud.npy, mask.npy)` but is currently untested on real data.
2. **Better fitting loss.** The current loss is a truncated Huber on SDF values.
   Move toward a *best-fit* objective: e.g. bidirectional/Chamfer surface
   distance, Eikonal/normal regularisation, or jointly optimising the alignment
   so the fit matches the observed surface rather than a fixed target SDF.
3. **Drop the canonical-pose assumption.** Today the object's pose is assumed
   and only a local similarity-ICP is solved (`alignment.py`), which needs a
   good init and diverges under large pose offsets. Add real **coarse
   alignment / object-pose estimation** *before* CSG refinement (global
   registration, symmetry handling, or a learned/LLM pose prior).
4. **Cluttered scenes + language grounding.** Identify and segment the target
   object in a cluttered scene, optionally via a **language prompt**, before
   fitting (extension).
5. **Topology from an LLM.** Have an LLM propose the tree topology *and* sensible
   normalised initial parameters (we optimise in a normalised space).
6. **Engineering.** Parallelise restarts (currently sequential), try
   second-order optimisers (LBFGS / Levenberg–Marquardt), and add mesh
   (marching-cubes) export for nicer visuals.

In-code `TODO:` comments mark where each of these hooks in.
