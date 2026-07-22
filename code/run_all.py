from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def resolve_script(filename: str) -> Path:
    """Locate a pipeline script in the repository's supported code locations."""
    candidates = [
        CODE / filename,
        CODE / "reproduction" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    # Last-resort recursive search, while rejecting duplicate ambiguous matches.
    matches = sorted(CODE.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        locations = "\n".join(f"  - {p}" for p in matches)
        raise RuntimeError(
            f"Multiple copies of {filename} were found; unable to choose safely:\n{locations}"
        )

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Required script not found: {filename}\nSearched:\n{searched}"
    )


CORE_FILENAMES = [
    "validate_package.py",
    "reproduce_persistence.py",
    "reproduce_persistent_models.py",
    "build_publication_outputs.py",
]

OPTIONAL_VALIDATOR_FILENAME = "validate_publication_outputs.py"


def run_script(path: Path, step_no: int, total_steps: int) -> None:
    print(f"\n[{step_no}/{total_steps}] Running {path.name}...", flush=True)
    runpy.run_path(str(path), run_name="__main__")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce all manuscript tables and figures. "
            "Publication-output validation is optional."
        )
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the optional publication-output validator after reproduction.",
    )
    args = parser.parse_args()

    try:
        steps = [resolve_script(name) for name in CORE_FILENAMES]
        if args.validate:
            steps.append(resolve_script(OPTIONAL_VALIDATOR_FILENAME))

        for idx, script in enumerate(steps, start=1):
            run_script(script, idx, len(steps))

    except Exception as exc:
        print(
            f"\nPipeline stopped because {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("\n" + "=" * 78)
    print("REPRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print("All manuscript tables and figures were generated successfully.")

    if not args.validate:
        print(
            "The publication-output validator was skipped by default. "
            "It can be run separately with:\n"
            "python -u code\\run_all.py --validate"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
