"""
download.py — Colab-friendly download of a single MVTec AD category.

LICENCE & DOWNLOAD INSTRUCTIONS
────────────────────────────────
MVTec AD is published by MVTec Software GmbH for non-commercial research only.
You must accept the licence agreement before downloading:

  https://www.mvtec.com/company/research/datasets/mvtec-ad

After accepting, per-category .tar.gz archives are available on that page.
Copy the URL (or download the file to Google Drive) and set ONE of:

  Option A — configs/default.yaml:
      download:
        archive_url: "https://<url-from-mvtec-page>/<category>.tar.gz"
        # OR
        archive_path: "/content/drive/MyDrive/mvtec/bottle.tar.gz"

  Option B — environment variables (preferred; never commit credentials):
      MVTEC_ARCHIVE_URL   or   MVTEC_ARCHIVE_PATH

No URL is hard-coded here because the links require licence acceptance
and may change without notice.
"""

import os
import tarfile
import urllib.request
from pathlib import Path


def get_category_data(cfg: dict) -> Path:
    """
    Ensure the category directory exists under cfg['data']['root'].

    Resolution order:
      1. Directory already present → skip.
      2. MVTEC_ARCHIVE_PATH env var → extract local/Drive archive.
      3. download.archive_path in config → extract local/Drive archive.
      4. MVTEC_ARCHIVE_URL env var → stream-download then extract.
      5. download.archive_url in config → stream-download then extract.
      6. Raise a clear error with instructions.

    Returns:
        Path to the extracted category directory.
    """
    root = Path(cfg["data"]["root"])
    category = cfg["data"]["category"]
    category_dir = root / category

    if category_dir.exists():
        print(f"[download] Already present: {category_dir}")
        _print_structure(category_dir)
        return category_dir

    dl = cfg.get("download", {})
    archive_path = os.environ.get("MVTEC_ARCHIVE_PATH") or dl.get("archive_path", "")
    archive_url = os.environ.get("MVTEC_ARCHIVE_URL") or dl.get("archive_url", "")

    if archive_path:
        _extract(Path(archive_path), root)
    elif archive_url:
        dest = root / f"{category}.tar.gz"
        root.mkdir(parents=True, exist_ok=True)
        _stream_download(archive_url, dest)
        _extract(dest, root)
        dest.unlink()
    else:
        raise ValueError(
            "No MVTec archive source configured.\n"
            "Set MVTEC_ARCHIVE_URL / MVTEC_ARCHIVE_PATH env vars, or\n"
            "set download.archive_url / download.archive_path in configs/default.yaml.\n"
            "See src/download.py for full instructions."
        )

    if not category_dir.exists():
        raise RuntimeError(
            f"Extraction finished but '{category_dir}' not found.\n"
            f"Ensure the archive contains a top-level folder named '{category}'."
        )

    _print_structure(category_dir)
    return category_dir


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading → {dest}")

    def _hook(count, block, total):
        if total > 0:
            print(f"\r  {min(count * block / total * 100, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def _extract(archive: Path, dest: Path) -> None:
    if not archive.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive}\n"
            "Upload it to Drive and update configs/default.yaml."
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] Extracting {archive.name} → {dest}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)
    print("[download] Done.")


def _print_structure(path: Path, max_depth: int = 2, _depth: int = 0) -> None:
    """Print directory tree up to max_depth for quick verification."""
    if _depth == 0:
        print(f"\n[structure] {path}")
    indent = "  " * _depth
    for child in sorted(path.iterdir()):
        print(f"{indent}{'├── ' if _depth else ''}{child.name}{'/' if child.is_dir() else ''}")
        if child.is_dir() and _depth < max_depth - 1:
            _print_structure(child, max_depth, _depth + 1)
