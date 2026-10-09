#!/usr/bin/env bash
# Compile main.tex to main.pdf (pdflatex, bibtex, pdflatex x2) and rebuild
# the arXiv upload archive.
# Usage: ./build.sh          build
#        ./build.sh clean    remove auxiliary files
set -euo pipefail

cd "$(dirname "$0")"
name=main
archive=arxiv_submission.tar.gz

# macOS metadata: Finder files and AppleDouble files on external drives
find . \( -name .DS_Store -o -name '._*' \) -type f -delete
xattr -cr . 2>/dev/null || true

if [[ "${1:-}" == "clean" ]]; then
  rm -f "$name".{aux,blg,log,out,toc}
  exit 0
fi

run_latex() {
  if ! pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" >/dev/null; then
    grep -n -A3 "^!" "$name.log" || tail -n 30 "$name.log"
    exit 1
  fi
}

run_latex
bibtex "$name" >/dev/null || { cat "$name.blg"; exit 1; }
run_latex
run_latex

if grep -q "undefined" "$name.log"; then
  echo "Warning: undefined references or citations:"
  grep "undefined" "$name.log"
fi

echo "Built $name.pdf"

# Archive for arXiv: source, bbl, bib and figures, without macOS metadata
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs \
  --exclude '._*' --exclude .DS_Store \
  --uid 0 --gid 0 --uname '' --gname '' \
  -czf "$archive" "$name.tex" "$name.bbl" references.bib figures/*.pdf

echo "Built $archive:"
tar -tzf "$archive" | sed 's/^/  /'
