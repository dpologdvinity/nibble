#!/usr/bin/env bash
# Usage: tools/convert_hf.sh <llama32-1b|tinyllama-1.1b>
# Downloads the HF repo (json, safetensors, tokenizer.model) at a pinned commit, verifies every file
# against the hashes the HF API reports for that commit, and converts it to an F16 GGUF with the
# pinned llama.cpp converter in a CPU-torch venv (third_party/py), offline. Skips steps whose output
# exists. Refuses repos whose config could make the converter run repo code (trust_remote_code).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: convert_hf.sh <llama32-1b|tinyllama-1.1b>}"
# REV: the commit whose files were checked against models/hf/<name>/ (main on 2026-10-18).
case "$NAME" in
  llama32-1b)
    REPO=unsloth/Llama-3.2-1B-Instruct; REV=5a8abab4a5d6f164389b1079fb721cfab8d7126c
    OUT="$ROOT/models/llama-3.2-1b-instruct-f16.gguf" ;;
  tinyllama-1.1b)
    REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0; REV=fe8a4ea1ffedaf415f4da2f062534de366a451e6
    OUT="$ROOT/models/tinyllama-1.1b-chat-f16.gguf" ;;
  *) echo "error: unknown model '$NAME' (known: llama32-1b tinyllama-1.1b)" >&2; exit 1 ;;
esac
if [ -f "$OUT" ]; then echo "skip $OUT"; exit 0; fi

PY="$ROOT/third_party/py"
if [ ! -x "$PY/bin/python" ]; then
  rm -rf "$PY"; python3 -m venv "$PY"
fi
if [ ! -f "$PY/.deps-ok" ]; then
  "$PY/bin/python" -m pip install -r "$ROOT/third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
  touch "$PY/.deps-ok"
fi

HF="$ROOT/models/hf/$NAME"
mkdir -p "$HF"
# Manifest lines "<file>\t<sha256|gitblob>\t<hex>" for the pinned commit; names are plain file names.
MANIFEST=$(curl -L --fail -s "https://huggingface.co/api/models/$REPO/revision/$REV?blobs=true" |
  "$PY/bin/python" -I -c '
import json, re, sys
m = json.load(sys.stdin)
if m.get("sha") != sys.argv[1]:
    sys.exit("error: API returned revision %r, want %s" % (m.get("sha"), sys.argv[1]))
for s in m["siblings"]:
    f = s["rfilename"]
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", f):
        continue
    if f.endswith(".json") or f.endswith(".safetensors") or f == "tokenizer.model":
        lfs = s.get("lfs")
        print("%s\t%s\t%s" % (f, "sha256" if lfs else "gitblob", lfs["sha256"] if lfs else s["blobId"]))' "$REV")
[ -n "$MANIFEST" ] || { echo "error: no files listed for $REPO@$REV" >&2; exit 1; }
while IFS=$'\t' read -r f _ _; do
  curl -L --fail -C - -o "$HF/$f" "https://huggingface.co/$REPO/resolve/$REV/$f"
done <<< "$MANIFEST"

# Verify every file: LFS files by sha256, the rest by git blob id. Then refuse configs that could
# load repo code: auto_map in config.json / tokenizer_config.json, or an unexpected tokenizer_class.
printf '%s\n' "$MANIFEST" | "$PY/bin/python" -I -c '
import hashlib, json, os, sys
d = sys.argv[1]
bad = False
for line in sys.stdin.read().splitlines():
    f, kind, want = line.split("\t")
    p = os.path.join(d, f)
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1(b"blob %d\0" % os.path.getsize(p))
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        print("error: %s: %s %s, want %s (delete it and re-run)" % (p, kind, h.hexdigest(), want), file=sys.stderr)
        bad = True
if bad:
    sys.exit(1)
for f in ("config.json", "tokenizer_config.json"):
    with open(os.path.join(d, f), encoding="utf-8") as fh:
        if "auto_map" in json.load(fh):
            sys.exit("error: %s has auto_map (repo code); refusing to convert" % f)
with open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8") as fh:
    cls = json.load(fh).get("tokenizer_class")
allowed = ("LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizer", "PreTrainedTokenizerFast")
if cls not in allowed:
    sys.exit("error: tokenizer_class %r not in %s; refusing to convert" % (cls, ", ".join(allowed)))
print("verified %s" % d)' "$HF"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY/bin/python" "$ROOT/third_party/llama.cpp/convert_hf_to_gguf.py" \
  "$HF" --outtype f16 --outfile "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
ls -l "$OUT"
