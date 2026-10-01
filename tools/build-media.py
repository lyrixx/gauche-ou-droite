# Construit les pages statiques de media/ à partir de tools/templates/ et tools/data/ :
# - media/nuage/index.html : le nuage des exemples ;
# - media/bench/index.html : le banc d'essai des classifieurs ;
# - media/slides/index.html : le lightning talk, depuis media/slides/src/ (deck.json + une slide par fichier).
#
# Pas de dépendance : python3 tools/build-media.py

import json
import re
from html import escape
from pathlib import Path

root = Path(__file__).resolve().parent.parent
templates = root / "tools/templates"
data = root / "tools/data"
media = root / "media"


def page(template: str) -> str:
    """Les gabarits commencent par <title>, les polices et un <style> : c'est l'en-tête du document."""
    head, body = template.split("</style>", 1)
    return (
        '<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'{head.strip()}\nbody {{ margin: 0; }}\n</style>\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n'
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"{path.relative_to(root)} ({len(content) // 1024} Ko)")


def build_nuage() -> None:
    t = (templates / "nuage.html").read_text()
    write(media / "nuage/index.html", page(t.replace('"__DATA__"', (data / "points.json").read_text())))


def build_bench() -> None:
    t = (templates / "bench.html").read_text()
    t = t.replace('"__DATA__"', (data / "bench.json").read_text()).replace('"__STAB__"', (data / "stability.json").read_text())
    write(media / "bench/index.html", page(t))


# Icônes Lucide utilisées par les slides (<x-icon name="…">).
ICONS = {
    "Check": '<path d="M20 6 9 17l-5-5"/>',
    "Wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
}


def icon(match: re.Match) -> str:
    name, style = match.group(1), match.group(2) or ""
    if name not in ICONS:
        raise SystemExit(f"Icône inconnue dans une slide : {name} (à ajouter dans ICONS)")
    return (f'<svg class="x-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="{style}">{ICONS[name]}</svg>')


def build_slides() -> None:
    src = media / "slides/src"
    deck = json.loads((src / "deck.json").read_text())
    slides = []
    for sid in deck["order"]:
        html = (src / "slides" / f"{sid}.html").read_text().strip()
        html = re.sub(r'<x-icon name="([^"]+)"(?: style="([^"]*)")?>\s*</x-icon>', icon, html)
        slides.append(html)
    fonts = "\n".join(f'<link rel="stylesheet" href="{escape(f["href"])}">' for f in deck["faces"].values() if "href" in f)
    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n' + fonts
    t = (templates / "slides.html").read_text()
    t = t.replace("__TITLE__", escape(deck["title"])).replace("__FONTS__", fonts).replace("__SLIDES__", "\n".join(slides))
    write(media / "slides/index.html", t)


build_nuage()
build_bench()
build_slides()
