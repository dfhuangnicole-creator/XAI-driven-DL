import argparse

from .common import SBERT_MODEL_DIR, ensure_sbert_model, sbert_model_ready


def parse_args():
    parser = argparse.ArgumentParser(description="Download the local SBERT model from Hugging Face.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite the local files even if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    already_ready = sbert_model_ready()
    model_dir = ensure_sbert_model(force_download=args.force)

    if already_ready and not args.force:
        print(f"SBERT model already available at: {model_dir}")
    else:
        print(f"SBERT model is ready at: {model_dir}")


if __name__ == "__main__":
    main()
