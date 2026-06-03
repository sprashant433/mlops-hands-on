import argparse

from mlops_lr.pipeline import run_pipeline
from mlops_lr.tuning_pipeline import run_tuning_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pipeline", "tuning"],
        default="pipeline",
    )
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline()
    elif args.mode == "tuning":
        run_tuning_pipeline()


if __name__ == "__main__":
    main()
