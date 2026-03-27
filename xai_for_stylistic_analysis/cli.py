import argparse
import importlib
import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / "tools"
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
OUTPUTS_DIR = ROOT_DIR / "outputs"


MODULE_MAP = {
    "fetch-sbert-model": "xai_for_stylistic_analysis.vae.fetch_sbert_model",
    "vae-preprocess": "xai_for_stylistic_analysis.vae.preprocess",
    "vae-train": "xai_for_stylistic_analysis.vae.train_VAE",
    "vae-search": "xai_for_stylistic_analysis.vae.train_cls",
    "vae-vote": "xai_for_stylistic_analysis.vae.vote_cls",
    "vae-final": "xai_for_stylistic_analysis.vae.final_cls",
    "vae-logistic": "xai_for_stylistic_analysis.vae.log_cls",
    "vae-perturb": "xai_for_stylistic_analysis.vae.pertubation",
    "vae-plot": "xai_for_stylistic_analysis.vae.plot",
    "vae-importance": "xai_for_stylistic_analysis.vae.feature_importance",
    "vae-sbert-search": "xai_for_stylistic_analysis.vae.train_cls_sbert",
    "vae-sbert-vote": "xai_for_stylistic_analysis.vae.vote_cls_sbert",
    "vae-sbert-final": "xai_for_stylistic_analysis.vae.final_cls_sbert",
}

SCRIPT_MAP = {
    "tfidf-preprocess": TOOLS_DIR / "tfidf_preprocessed.py",
    "tfidf-vote": TOOLS_DIR / "tfidf_vote.py",
}


def run_script(script_path: Path):
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    runpy.run_path(str(script_path), run_name="__main__")


def run_module(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module {module_name} does not expose a main() function.")
    module.main()


def print_check():
    checks = [
        ("Raw input directory", DATA_DIR / "raw"),
        ("Processed TF-IDF data", DATA_DIR / "processed" / "tfidf" / "processed_TFIDF_final.json"),
        ("TF-IDF feature names", DATA_DIR / "processed" / "tfidf" / "feature_names.json"),
        ("Processed VAE data", DATA_DIR / "processed" / "vae" / "processed_final.json"),
        ("VAE weights", OUTPUTS_DIR / "vae" / "vae_model.pth"),
        ("Local SBERT model", ASSETS_DIR / "models" / "all-mpnet-base-v2"),
    ]
    print("Project check:\n")
    for label, path in checks:
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {label}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Unified project entrypoint.")
    parser.add_argument(
        "command",
        choices=["check", *MODULE_MAP.keys(), *SCRIPT_MAP.keys()],
        help="What to run.",
    )
    args, remaining = parser.parse_known_args()

    if args.command == "check":
        print_check()
        return

    if args.command in MODULE_MAP:
        sys.argv = [MODULE_MAP[args.command], *remaining]
        run_module(MODULE_MAP[args.command])
        return

    script_path = SCRIPT_MAP[args.command]
    sys.argv = [str(script_path), *remaining]
    run_script(script_path)


if __name__ == "__main__":
    main()
