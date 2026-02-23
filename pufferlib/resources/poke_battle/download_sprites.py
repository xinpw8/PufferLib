#!/usr/bin/env python3
"""Download Gen 1 Pokemon sprites from Pokemon Showdown CDN."""

from pathlib import Path
import os
import sys
import urllib.request

BASE_URL = "https://play.pokemonshowdown.com/sprites"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
FRONT_DIR = SCRIPT_DIR / "sprites" / "gen1"
BACK_DIR = SCRIPT_DIR / "sprites" / "gen1-back"

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib.ocean.poke_battle.poke_battle import LEGAL_SPECIES_IDS, SPECIES_NAMES


def to_showdown_id(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def species_slugs() -> list[str]:
    return [to_showdown_id(SPECIES_NAMES[s]) for s in LEGAL_SPECIES_IDS]


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)


def download():
    FRONT_DIR.mkdir(parents=True, exist_ok=True)
    BACK_DIR.mkdir(parents=True, exist_ok=True)

    slugs = species_slugs()
    failed = []

    for name in slugs:
        for subdir, dest_dir in [("gen1", FRONT_DIR), ("gen1-back", BACK_DIR)]:
            url = f"{BASE_URL}/{subdir}/{name}.png"
            dest = dest_dir / f"{name}.png"
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  skip {dest} (exists)")
                continue
            print(f"  downloading {url}")
            try:
                fetch(url, dest)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append(url)

    print(f"\nDone: requested {len(slugs)} species x2 views ({len(slugs) * 2} files).")
    if failed:
        print(f"Failures: {len(failed)}")
        for url in failed:
            print(f"  {url}")
    else:
        print("No download failures.")


if __name__ == "__main__":
    download()
