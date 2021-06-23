import os

BASE_PATH = os.path.dirname(__file__)
CHAR_FPATH = os.path.join(BASE_PATH, 'ko_char.txt')
with open(CHAR_FPATH, "r", encoding="utf-8-sig") as f:
    characters = f.read().replace('\n', '')

ko = characters
print(f"ko char. len : {len(ko)}")
print(ko)
