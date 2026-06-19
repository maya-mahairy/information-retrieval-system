import argparse

from services.dataset_preparation_service import prepare_all_data


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess the IR dataset.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 1000",
    )

    args = parser.parse_args()

    prepare_all_data(limit=args.limit)


if __name__ == "__main__":
    main()