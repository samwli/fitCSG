# fitCSG

Fit a **Constructive Solid Geometry (CSG) tree of signed-distance primitives**
to a 3D object, as a compact, interpretable, editable, grasp-aware shape
abstraction (the work behind the *CSGGrasp* submission).

An LLM (or a human) proposes a **hypothesis**: a CSG tree of shapes + boolean
ops **and sensible initial parameters** (center / size / rotation), authored in a
normalised ~unit-cube space. That hypothesis is then **optimised** so its SDF
matches a target derived from an observed object — the parameters refine from a
plausible guess to a tight fit of the specific instance. The starting point is a
meaningful hypothesis, **not** random initialisation.

![CSG fitting a point cloud](assets/fit_demo.gif)

*Above: the data-free demo. An LLM-style **abstract mug hypothesis**
(`examples/mug_init.json`, colour — taller/thinner, tilted handle) is optimised
until it fits the **actual instance** (`examples/mug.json`, grey point cloud).
This mirrors the intended workflow — start from a plausible hypothesis, not from
random — and the parameters refine to match the specific object.*

> **Status.** This was research code that was buggy and partly faked-to-work; it
> was rebuilt in June 2026 into a clean, *correct* implementation of the idea
> (`fitcsg/` package + tests), and the LLM hypothesis generator (previously on a
> separate branch) was merged in and wired up. It runs CPU-only; a GPU just
> speeds up fitting. The synthetic demo and the generate→fit flow (with cached
> hypotheses) are reproducible; the live LLM call and the real-data path are
> implemented but unexercised here (no API key / captured data — see TODOs).

## Pipeline

```
object name --> LLM --> CSG hypothesis (tree + init params)  ─┐   llmhypothesis/
                                                              ├─> optimise CSG leaf params ─> fitted tree ─> part segmentation
RGB image --(external)--> point cloud --> target SDF  ────────┘   fitcsg/
```

This repo owns the whole loop end-to-end:

* **LLM hypothesis generation** (`llmhypothesis/`) — object name → CSG tree +
  initial parameters, via TypeChat + a TypeScript schema;
* **CSG → SDF**, **point cloud → target SDF**, the **optimisation loop**, and
  **part segmentation** (`fitcsg/`).

Only the `RGB → point cloud` front-end (depth back-projection / reconstruction)
lives outside the repo. See [`llmhypothesis/README.md`](llmhypothesis/README.md)
for the generator and [its setup](#llm-hypothesis-generation) below.

## The idea / intuition

Most everyday objects are well approximated by a few **simple analytic shapes**
combined with boolean operations. Instead of a dense mesh or neural field, we
describe an object as a small **CSG program** — a tree whose leaves are
primitives (cylinder, box, sphere, …) and whose internal nodes are
`union` / `intersection` / `subtraction`. Each primitive has an exact
signed-distance function (SDF), and the boolean ops compose SDFs analytically, so
the whole object has a differentiable SDF we can fit by gradient descent.

This representation is **compact** (a handful of numbers), **interpretable**
(you can read off "a hollow cylinder with a handle"), and **editable** (change a
radius, move a part) — which is exactly what's useful for downstream grasping.

**Worked example — a mug** (`examples/mug.json`, the demo target):

```
union(
    subtraction(              # hollow body:
        cylinder(body),       #   a solid outer cylinder
        cylinder(cavity)      #   minus a slightly smaller inner cylinder -> the cup hole
    ),
    torus(handle)             # plus a torus for the handle
)
```

The art is choosing *which* primitives and ops express an object (the topology),
then letting optimisation dial in the exact parameters. An LLM is well suited to
the first part ("a mug is a hollow cylinder with a ring handle"); optimisation
handles the second.

## How it works (intended workflow)

1. **Hypothesis (LLM).** An LLM proposes a CSG tree — shapes, boolean ops, *and*
   initial parameters — in a normalised ~[-1, 1] unit cube. This is implemented
   in [`llmhypothesis/`](llmhypothesis/) (`python generate.py --object mug`) and
   emits a JSON file; cached examples live in `llmhypothesis/csg_hypotheses/` and
   a hand-authored one in `examples/mug_init.json`. Good initial params matter:
   we optimise in the normalised space, so a roughly-right guess converges far
   more reliably. `fitcsg.parse_tree` loads the generator's schema directly.
2. **Observation.** The object's point cloud is normalised to the same scale.
3. **Coarse alignment** *(TODO — not yet implemented)*. The hypothesis and the
   observation have an unknown relative pose; a coarse alignment / pose estimate
   would bring them into rough correspondence. Today this step is skipped and the
   examples are authored already-aligned (`alignment.py` only has a *local*
   similarity-ICP placeholder, which needs a good initial pose).
4. **Target SDF.** Build a target SDF from the (aligned) observation
   (`target.py`); or, with no external data, sample one from a known tree
   (`synthetic.py`).
5. **Optimise (this is the core).** Refine the hypothesis's continuous leaf
   parameters to minimise the SDF loss (`optimize.py`). The **topology is fixed**;
   only the parameters move. Two options keep this anchored to the meaningful
   hypothesis: `reg_weight` (an L2 pull back toward the initial params) and
   `coarse_to_fine` (place part *centers* first, then unfreeze all params).
6. **Output + part segmentation.** A fitted CSG tree — compact, interpretable,
   editable. `fitcsg.segment_point_cloud` then labels each observed point by the
   fitted part it belongs to (body vs. handle vs. …) for downstream grasping.

The demo GIF above is exactly steps 1→5 with a synthetic instance: start from the
abstract `mug_init.json` hypothesis (already coarsely aligned), optimise onto the
`mug.json` instance.

## Shape primitives

All primitives share `center` (world origin) and `rotation` (XYZ Euler degrees);
the shape-specific params are listed below. Defined in `fitcsg/primitives.py`;
canonical frame is the origin, aligned to local **+Z** where an axis matters.

| shape       | extra params          | good for / notes |
|-------------|-----------------------|------------------|
| `sphere`    | `radius`              | balls, blobs, rounded caps. Exact SDF. |
| `ellipsoid` | `size` (3)            | squashed/elongated spheres (eggs, heads). *Approximate SDF* (no closed form). |
| `box`       | `size` (3)            | cuboids, slabs, frames, table-tops. Exact SDF. |
| `cylinder`  | `radius`, `height`    | cups, cans, tubes, legs, rods. Axis = +Z. Exact SDF. |
| `cone`      | `radius`, `height`    | funnels, tips, tapers (apex at +Z). Exact capped-cone SDF. |
| `torus`     | `radius` (major), `tube` (minor) | handles, rings, rims, donuts. Lies in the XY-plane. Exact SDF. |
| `capsule`   | `radius`, `height`    | rounded rods / limbs / grips (a cylinder with hemispherical caps). Axis = +Z. Exact SDF. |

Combine them with `union` (∪, `min`), `intersection` (∩, `max`) and
`subtraction` (−, carve one out of another) — optionally with a `smooth` blend
radius for soft joins.

### Future primitives to consider

Useful additions for covering more everyday objects (each is a small canonical
SDF + a one-line registry entry in `PRIMITIVES`):

* **rounded box** — boxes with filleted edges (most manufactured objects);
* **capped / partial torus** — a handle arc rather than a full ring;
* **n-gon prism / wedge** — hex bolts, pyramids, ramps, triangular cross-sections;
* **half-space / plane** — slice a shape flat (table cuts, flat bases);
* **rounded cylinder** — cans/bottles with rounded top edges;
* **superquadric / superellipsoid** — one shape that smoothly interpolates
  box↔sphere↔cylinder via 2 exponents (very expressive for organic + manufactured
  parts; a strong candidate if you want fewer leaves per object).

## Onboarding (new here? start with this)

If you've never seen this project before:

1. **Run the demo** (no data, no GPU needed) and watch it fit:
   `python scripts/fit_demo.py --tree examples/mug.json --init_tree examples/mug_init.json --outdir demo_out`
2. **Read in this order:** this README → `fitcsg/csg.py` (what a tree *is*) →
   `fitcsg/primitives.py` (the shapes) → `fitcsg/optimize.py` (the fitting loop)
   → `scripts/fit_demo.py` (how it's all wired together).
3. **Mental model:** a "hypothesis" / "tree" is just a JSON CSG tree (see
   `examples/`). `parse_tree` turns it into objects; `evaluate` gives its SDF;
   `fit` moves its leaf params to match a target SDF.
4. **Common tasks:** author a hypothesis → copy an `examples/*.json` and edit
   params (schema under *Conventions*). Add a new primitive → write a canonical
   SDF and register it in `PRIMITIVES` (`fitcsg/primitives.py`). Implement coarse
   alignment → start from `fitcsg/alignment.py` and the hook in `target.py`.
5. Every module has a top docstring explaining what it does and what's missing;
   `TODO:` comments mark the open roadmap items in-place.

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
  optimize.py     fitting loop: truncated-Huber loss, cosine LR, restarts, init-reg, coarse->fine
  segment.py      label observed points by fitted CSG part (for grasping)
  visualize.py    Graphviz tree + matplotlib SDF/fit rendering + GIF assembly
  random_tree.py  random tree generation (smoke tests)
scripts/
  visualize_tree.py   render a tree's graph and/or SDF
  fit.py              fit a tree to a synthetic or real target
  fit_demo.py         animated fit -> GIF (the visual smoke test)
llmhypothesis/        LLM CSG-hypothesis generator (TypeChat + TS schema); see its README
  generate.py         object name -> CSG hypothesis JSON
  csgschema.ts        the generator-side schema (ingested by fitcsg.parse_tree)
  prompt.md           the generation prompt
  csg_hypotheses/     cached example hypotheses (mug, knife, screwdriver, ...)
  TypeChat/           git submodule (run: git submodule update --init --recursive)
examples/
  sunglasses.json     original hand-authored example (converted to new schema)
  mug.json            demo *instance*: cylinder - cavity + torus handle
  mug_init.json       abstract mug *hypothesis* (LLM-style start for the demo)
tests/                pytest suite (run: pytest)
```

## Installation

```bash
conda env create -f environment.yml
conda activate fitcsg
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build

# only needed to *generate* new hypotheses with the LLM (not for fitting):
git submodule update --init --recursive    # pulls llmhypothesis/TypeChat
```

Fitting and the demo need only `torch`/`numpy`/`matplotlib` (the `fitcsg/`
package). The LLM generator additionally needs the TypeChat submodule, a
TypeScript compiler, and an API key — see
[LLM hypothesis generation](#llm-hypothesis-generation).

## Quickstart

```bash
# Visualise a tree's SDF (no GPU)
python scripts/visualize_tree.py --tree examples/mug.json --save mug.png

# Self-contained fit: optimise an abstract hypothesis onto a synthetic instance
python scripts/fit.py --tree examples/mug.json --init_tree examples/mug_init.json
# (omit --init_tree to instead randomise params and test recovery, with restarts)
python scripts/fit.py --tree examples/mug.json --restarts 4

# Fit an actual cached LLM hypothesis (loads the generator schema directly),
# anchored to the proposal via init-regularisation + coarse-to-fine
python scripts/fit.py --tree examples/mug.json \
    --init_tree llmhypothesis/csg_hypotheses/csg_mug_1.json \
    --reg_weight 1e-3 --coarse_to_fine

# Generate a fresh hypothesis with the LLM (needs submodule + API key)
python llmhypothesis/generate.py --object mug

# Animated GIF: an abstract mug hypothesis optimised onto the actual instance
python scripts/fit_demo.py --tree examples/mug.json \
    --init_tree examples/mug_init.json --num_steps 500 --outdir demo_out

# Fit to a real observation (needs your own data; see TODOs)
python scripts/fit.py --target files --tree examples/mug.json \
    --pc pointcloud.npy --mask mask.npy

# Tests
pytest
```

## Tests

`pytest` (20 tests) covers the correctness claims above so the next person can
refactor safely:

* `test_transforms.py` — rotation matrices are orthonormal with `det=1`;
  world→local preserves distances.
* `test_primitives.py` — exact sphere/box distances; every primitive encloses a
  non-empty solid and reports far points as outside; box rotation-equivariance;
  size-sign invariance.
* `test_csg.py` — union/intersection/subtraction signs; overlapping-sphere
  volumes; JSON round-trip; **LLM schema loading** (`Prism`→box, `part`→name,
  `axis` direction → Euler `rotation`); colour output shape.
* `test_optimize.py` — the synthetic fit reduces loss by >2× and converges; the
  **coarse-to-fine + init-regularisation** path runs and limits drift.
* `test_segment.py` — every observed point gets one part label and the parts are
  populated.
* `test_random_tree.py` — random trees parse and yield a non-empty solid.

## LLM hypothesis generation

The front of the pipeline lives in [`llmhypothesis/`](llmhypothesis/) and turns
an object name into a CSG tree (shapes + initial params) using
[TypeChat](https://github.com/microsoft/TypeChat) to constrain the LLM to the
`csgschema.ts` schema. Full instructions are in
[`llmhypothesis/README.md`](llmhypothesis/README.md); in short:

```bash
git submodule update --init --recursive    # TypeChat
export OPENAI_API_KEY=sk-...
python llmhypothesis/generate.py --object mug          # -> llmhypothesis/csg_hypotheses/csg_mug_1.json
python scripts/fit.py --tree examples/mug.json \
    --init_tree llmhypothesis/csg_hypotheses/csg_mug_1.json --reg_weight 1e-3 --coarse_to_fine
```

The generator emits the *legacy* schema (`operation`/`type`/`sizes`/`axis`/
`part`); `fitcsg.parse_tree` maps it to the canonical schema automatically
(`Prism`→`box`, `type` suffix stripped, `part`→`name`, `axis` direction →
Euler `rotation`). Cached hypotheses ship in `llmhypothesis/csg_hypotheses/`, so
the generate→fit flow can be exercised end-to-end without an API key by using
those files as `--init_tree`.

> The generator was developed on a separate branch; it has been merged here and
> wired up (schema ingestion + CLI + docs). The fit side is fully tested; the
> live LLM call itself requires the submodule, a TypeScript compiler and an API
> key, so it has not been re-run in this environment.

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
  `{"shape", "name", "params": {...}}` for leaves. The legacy / LLM schema
  (`operation`/`type`/`sizes`/`axis`/`part`) also loads: `Prism`→`box`, `type`
  suffix stripped, `part`→`name`, and the `axis` direction is converted to the
  canonical Euler `rotation` for axis-aligned shapes.

## What the rebuild fixed

Brief provenance (the original bugs): rotation was ignored for some primitives
and inconsistent for others; the JSON `rotation` key didn't match the `axis` key
the SDFs read (the shipped example would have crashed); `random_tree_utils.py`
couldn't import; leaf lookup broke for ≥10 instances of a shape; viz colours
were non-deterministic. All fixed, plus a robust loss, restarts, more primitives
(torus/capsule), and a test suite.

The rebuild also **consolidated the divergent branches** into one `main`: the
LLM hypothesis generator from the `llmhypothesis` branch was merged in (history
preserved), and that branch's still-valuable research intentions were ported
into the package rather than left on dead code — init-parameter regularisation
and coarse-to-fine optimisation (`optimize.py`), and post-fit part segmentation
(`segment.py`).

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
5. **Topology from an LLM** *(implemented — see `llmhypothesis/`)*. An LLM
   proposes the tree topology *and* normalised initial parameters; the output
   feeds straight into `--init_tree`. Remaining work: validate hypothesis quality
   at scale, constrain/standardise the proposed pose to help coarse alignment
   (#3), and optionally migrate the generator to emit the canonical schema
   directly instead of relying on the parser's legacy mapping.
6. **Engineering.** Parallelise restarts (currently sequential), try
   second-order optimisers (LBFGS / Levenberg–Marquardt), and add mesh
   (marching-cubes) export for nicer visuals.

In-code `TODO:` comments mark where each of these hooks in.
