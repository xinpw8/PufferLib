#!/usr/bin/env python3
"""Download Gen 1 Pokemon sprites from Pokemon Showdown CDN."""

import os
import urllib.request

SPECIES = [
    "tauros", "chansey", "snorlax", "alakazam", "exeggutor",
    "starmie", "gengar", "jynx", "zapdos", "rhydon",
    "cloyster", "golem", "lapras", "slowbro", "jolteon",
    "persian", "hypno", "articuno", "dragonite", "machamp",
]

BASE_URL = "https://play.pokemonshowdown.com/sprites"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(SCRIPT_DIR, "sprites", "gen1")
BACK_DIR = os.path.join(SCRIPT_DIR, "sprites", "gen1-back")

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)


def download():
    os.makedirs(FRONT_DIR, exist_ok=True)
    os.makedirs(BACK_DIR, exist_ok=True)

    for name in SPECIES:
        for subdir, dest_dir in [("gen1", FRONT_DIR), ("gen1-back", BACK_DIR)]:
            url = f"{BASE_URL}/{subdir}/{name}.png"
            dest = os.path.join(dest_dir, f"{name}.png")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"  skip {dest} (exists)")
                continue
            print(f"  downloading {url}")
            try:
                fetch(url, dest)
            except Exception as e:
                print(f"  FAILED: {e}")


if __name__ == "__main__":
    download()
