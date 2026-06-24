"""Generate a CSG-tree hypothesis for an object with an LLM.

This is the *front* of the pipeline: given an object name (e.g. ``mug``), an LLM
proposes a Constructive-Solid-Geometry tree -- shapes **and** plausible initial
parameters in a roughly unit-cube space -- constrained by ``csgschema.ts`` (via
TypeChat) and ``prompt.md``.

The resulting JSON is consumed directly by the fitting half of the repo::

    python scripts/fit.py --tree <target>.json \
        --init_tree llmhypothesis/csg_hypotheses/csg_<object>_1.json \
        --reg_weight 1e-3 --coarse_to_fine

``fitcsg.parse_tree`` ingests this (legacy) schema natively -- ``Prism`` maps to
``box``, the numeric ``type`` suffix is stripped, ``part`` becomes the leaf name,
and the ``axis`` direction is converted to the canonical Euler ``rotation``.

Setup (see README.md): initialise the TypeChat submodule, install its deps + a
TypeScript compiler, and provide an API key via ``--api_key`` or the
``OPENAI_API_KEY`` environment variable.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_typechat():
    """Import TypeChat lazily with a helpful message if the submodule is missing.

    Keeping this out of module scope means ``--help`` and argument validation
    work even before the submodule / its dependencies are installed; only the
    actual generation needs TypeChat.
    """
    sys.path.insert(0, HERE)
    try:
        from TypeChat.typechat.typechat import TypeChat
    except ImportError as exc:
        raise SystemExit(
            "Could not import TypeChat. Initialise the submodule and install its "
            "dependencies first:\n"
            "    git submodule update --init --recursive\n"
            "    pip install -r llmhypothesis/TypeChat/requirements.txt  # see TypeChat docs\n"
            f"(original error: {exc})"
        )
    return TypeChat


def generate_tree(tns, object_name, prompt_path, out_path, hypothesis=None):
    """Translate one prompt into a schema-valid CSG JSON and save it."""
    with open(prompt_path) as f:
        prompt = f.read()
    prompt = prompt.replace("<ObjectName>", object_name)

    request = [{"role": "user", "content": prompt}]
    if hypothesis:
        request.append({"role": "assistant", "content": hypothesis})
        request.append(
            {
                "role": "user",
                "content": (
                    "Real world objects have diverse instances. Please generate "
                    f"another tree representing a geometrically different instance "
                    f"of the {object_name} object."
                ),
            }
        )

    response = tns.translate(request, image=None, return_query=False)
    if not response.success:
        print(response.error)
        return None

    csg = json.loads(str(response.data).replace("'", '"'))
    with open(out_path, "w") as f:
        f.write(json.dumps(csg, indent=4))
    print(f"wrote {out_path}")
    return json.dumps(csg, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate CSG hypotheses with an LLM")
    parser.add_argument("--object", required=True, help="object name, e.g. 'mug'")
    parser.add_argument("--model", default="o3-mini", help="o3-mini (paper default); gpt-4o is a cheap alternative")
    parser.add_argument("--num_variants", type=int, default=1, help="how many distinct instances to generate")
    parser.add_argument("--outdir", default=os.path.join(HERE, "csg_hypotheses"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--prompt", default=os.path.join(HERE, "prompt.md"))
    parser.add_argument("--schema", default=os.path.join(HERE, "csgschema.ts"))
    args = parser.parse_args()

    if not args.api_key:
        parser.error("no API key: pass --api_key or set OPENAI_API_KEY")
    os.makedirs(args.outdir, exist_ok=True)

    TypeChat = _import_typechat()
    ts = TypeChat()
    ts.createLanguageModel(
        model=args.model,
        api_key=args.api_key,
        org_key=None,
        use_json_mode=True,
        user_mode_only=True,
    )
    ts.loadSchema(args.schema)
    tns = ts.createJsonTranslator(
        name="CSGResponse", basedir=os.path.join(HERE, "TypeChat", "typechat", "schemas")
    )

    previous = None
    for variant in range(1, args.num_variants + 1):
        out_path = os.path.join(args.outdir, f"csg_{args.object}_{variant}.json")
        previous = generate_tree(tns, args.object, args.prompt, out_path, hypothesis=previous)
        if previous is None:
            break


if __name__ == "__main__":
    main()
