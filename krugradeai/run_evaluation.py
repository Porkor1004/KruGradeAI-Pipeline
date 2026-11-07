"""Evaluate an external prompt and teacher rubric without optimization."""
import argparse
from krugradeai.model_routing import ModelRoutingError
import json
from krugradeai.evaluation import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.config, live=args.live, dry_run=args.dry_run)
    except ModelRoutingError as error:
        parser.exit(1, f"Model routing failed: {error}\n")
    except Exception as error:
        parser.exit(1, f"Evaluation refused/incomplete ({type(error).__name__}); private input withheld.\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "metrics", "metrics_by_label_origin", "records", "maximum_calls", "runtime")}, indent=2))
    return 1 if result.get("status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
