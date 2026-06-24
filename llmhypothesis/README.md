# LLM hypothesis generation

This is the **front** of the pipeline. Given an object name, an LLM proposes a
Constructive-Solid-Geometry (CSG) tree — shapes **and** plausible initial
parameters in a roughly unit-cube space — which the fitting half of the repo
then optimises onto an observed instance.

```
LLM (this folder)                         fitcsg (../fitcsg)
object name ── prompt.md + csgschema.ts ──► csg_<object>_N.json ──► parse_tree ──► fit
                                                                    (--init_tree)
```

The generated JSON is consumed **directly** by the optimiser; `fitcsg.parse_tree`
ingests this schema natively (see "Schema" below), so no manual conversion is
needed.

## Setup

1. **Submodule** (TypeChat is a git submodule):
   ```bash
   git submodule update --init --recursive
   ```
2. **Dependencies**: install the TypeChat package's requirements and a
   TypeScript compiler (`tsc`) — TypeChat uses it to validate that the LLM
   output matches `csgschema.ts`. See `TypeChat/`'s own docs.
3. **API key**: export it (preferred) or pass `--api_key`:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Generate

```bash
# one hypothesis
python generate.py --object mug

# two geometrically distinct instances, cheaper model
python generate.py --object wire_cutters --num_variants 2 --model gpt-4o
```

`gpt-4o` is cheap (good for debugging); `o1-preview` is stronger but expensive.
Output is written to `csg_hypotheses/csg_<object>_<n>.json`.

## Fit a generated hypothesis

```bash
cd ..
python scripts/fit.py --tree examples/mug.json \
    --init_tree llmhypothesis/csg_hypotheses/csg_mug_1.json \
    --reg_weight 1e-3 --coarse_to_fine
```

(Here `--tree` stands in as the target instance; in a real run the target comes
from an observed point cloud via `--target files`.)

## Cached hypotheses

`csg_hypotheses/` ships example outputs (mug, knife, screwdriver, sunglasses,
wire_cutters) so the pipeline can be exercised without an API key. They all load
and fit with the commands above.

## Prompts

- `prompt.md` — the main prompt (the `<ObjectName>` placeholder is substituted).
- `prompt-old.md` — an earlier version, kept for reference.
- The follow-up "generate a different instance" prompt lives in `generate.py`.

## Schema

`csgschema.ts` is the generator-side schema TypeChat enforces. It is the
**legacy** schema (`operation`/`type`/`sizes`/`axis`/`part`); `fitcsg.parse_tree`
maps it to the canonical internal schema:

| LLM schema            | fitcsg canonical                          |
|-----------------------|-------------------------------------------|
| `operation`           | `op` (lower-cased)                        |
| `type: "Cylinder0"`   | `shape: "cylinder"` (suffix stripped)     |
| `type: "Prism0"`      | `shape: "box"`                            |
| `sizes`               | `size`                                    |
| `axis` (direction)    | `rotation` (Euler degrees, axis shapes)   |
| `part`                | leaf `name`                               |

If you add a shape here, also register it in `fitcsg/primitives.py` and the
alias tables in `fitcsg/csg.py`.

## Visualise

```bash
python viz.py csg_hypotheses/csg_mug_1.json
```

The standalone `viz.py` here is the original quick viewer and can be flaky.
Prefer the maintained renderer in the package:

```bash
cd ..
python scripts/visualize_tree.py --tree llmhypothesis/csg_hypotheses/csg_mug_1.json
```
