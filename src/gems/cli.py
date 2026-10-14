"""Command line: `gems run --config <toml>` and `gems validate <tif>`."""

import argparse
import sys

from gems.runner import DEFAULTS, InvalidSubmissionError, run_experiment
from gems.submission import validate_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gems", description="GEMS Prize fault probability maps.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an experiment from a TOML config.")
    run.add_argument("--config", required=True, help="Path to a TOML config, e.g. configs/v1.toml")

    val = sub.add_parser("validate", help="Check a GeoTIFF against the submission format.")
    val.add_argument("tif", help="GeoTIFF to check")
    val.add_argument("--features", default=DEFAULTS["data"]["features"], help="Reference grid raster")
    val.add_argument("--labels", default=DEFAULTS["data"]["labels"], help="Label raster defining the study area")

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            result = run_experiment(args.config)
        except InvalidSubmissionError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Submission: {result.submission_path}")
        return 0

    problems = validate_submission(args.tif, args.features, args.labels)
    if not problems:
        print(f"{args.tif}: valid submission.")
        return 0
    print(f"{args.tif}: {len(problems)} problem(s):")
    for p in problems:
        print(f"- {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
