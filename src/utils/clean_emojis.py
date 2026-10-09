# src/utils/clean_emojis.py
import os
import re

emoji_pattern = re.compile(
    r"[\U00010000-\U0010ffff]"
    r"|[\u2600-\u27bf]"
    r"|[\u2300-\u23ff]"
    r"|[\u2b50-\u2b55]"
    r"|[\u203c-\u2049]"
    r"|[\u25aa-\u25fe]"
    r"|[\u00a9\u00ae]"
    r"|[\u2122\u2139]"
    r"|[\u2934-\u2935]"
    r"|[\u3297\u3299]"
    r"|[\u3030\u303d]"
    r"|[\ufe0e\ufe0f]"
)


def clean_file(fpath: str):
    if not os.path.exists(fpath):
        return
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if emoji_pattern.search(content):
        # Nettoyer page_icon="..."
        cleaned = re.sub(r"page_icon=[\"\'][^\"\']*[\"\']\s*,\s*", "", content)
        cleaned = re.sub(r",\s*page_icon=[\"\'][^\"\']*[\"\']", "", cleaned)
        # Supprimer les emojis
        cleaned = emoji_pattern.sub("", cleaned)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"Cleaned emojis from {fpath}")


def clean_directory(dir_path: str):
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith((".py", ".sql", ".md")):
                clean_file(os.path.join(root, f))


if __name__ == "__main__":
    for d in ["src", "dags", "perso"]:
        if os.path.exists(d):
            clean_directory(d)
    print("Nettoyage des emojis terminé.")
