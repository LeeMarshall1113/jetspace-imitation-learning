#!/usr/bin/env bash
# Build the paper, and refuse to produce a clean PDF while decisions are open.
#
#   bash paper/build.sh            build
#   bash paper/build.sh --figures  regenerate figures from cache/*.json first
#   bash paper/build.sh --check    report TODOs and stale numbers, do not build
set -uo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--figures" ] || [ "${2:-}" = "--figures" ]; then
    echo "### regenerating figures from the result JSONs"
    (cd .. && python scripts/make_figures.py --out paper/figures)
    echo
fi

echo "### open decisions"
n=$(grep -c '\\TODO' main.tex 2>/dev/null || echo 0)
b=$(grep -c '\\TODO' references.bib 2>/dev/null || echo 0)
echo "  main.tex       ${n} TODO"
echo "  references.bib ${b} unverified entries"
if [ "$n" -gt 0 ]; then
    echo
    grep -n '\\TODO' main.tex | sed 's/\\TODO{/ -> /' | cut -c1-100 | head -12
    [ "$n" -gt 12 ] && echo "  ... and $((n - 12)) more"
fi

if [ "${1:-}" = "--check" ]; then
    echo
    echo "### citation hygiene"
    echo "  Every entry in references.bib was assembled from automated"
    echo "  literature search. Open each on arXiv once and delete what you"
    echo "  cannot confirm. A wrong citation is the cheapest way to lose a"
    echo "  reviewer."
    exit 0
fi

command -v pdflatex >/dev/null 2>&1 || {
    echo
    echo "pdflatex not found. Either install texlive, or use Overleaf:"
    echo "  upload main.tex, references.bib and figures/, and pick the venue"
    echo "  style there. TMLR: https://github.com/JmlrOrg/tmlr-style-file"
    exit 1
}

echo
echo "### building"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 \
    || { echo "pdflatex failed; rerun without the redirect to see why"; exit 1; }
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
echo "  main.pdf  ($(du -h main.pdf 2>/dev/null | cut -f1))"
pages=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')
[ -n "${pages:-}" ] && echo "  ${pages} pages"
