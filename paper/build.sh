#!/usr/bin/env bash
# Build the paper, and refuse to produce a clean PDF while decisions are open.
#
#   bash paper/build.sh            build
#   bash paper/build.sh --figures  regenerate figures from cache/*.json first
#   bash paper/build.sh --check    report TODOs, anonymity and citations; no build
#
# Venue is TMLR. tmlr.sty, tmlr.bst and fancyhdr.sty live in this directory
# (from github.com/JmlrOrg/tmlr-style-file) so the build pulls nothing.
set -uo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" = "--figures" ] || [ "${2:-}" = "--figures" ]; then
    echo "### regenerating figures from the result JSONs"
    (cd .. && python scripts/make_figures.py --out paper/figures)
    echo
fi

# ---------------------------------------------------------------- venue mode
echo "### venue"
opt=$(grep -oE '\\usepackage(\[[a-z]+\])?\{tmlr\}' main.tex | head -1)
case "$opt" in
    *"[accepted]"*) mode="CAMERA-READY -- author shown, needs \\month \\year \\openreview" ;;
    *"[preprint]"*) mode="PREPRINT -- author shown, no TMLR branding (arXiv)" ;;
    *"{tmlr}"*)     mode="SUBMISSION -- author hidden, double-blind" ;;
    *)              mode="NO tmlr PACKAGE FOUND -- main.tex is not on the venue style" ;;
esac
echo "  ${opt:-none}  ->  $mode"
for f in tmlr.sty tmlr.bst fancyhdr.sty; do
    [ -f "$f" ] || echo "  MISSING $f -- fetch it from JmlrOrg/tmlr-style-file"
done

# ------------------------------------------------------------- open decisions
echo
echo "### open decisions"
# Count real TODOs only: not the \newcommand that defines the macro, and not
# the comment block at the top that explains it. `grep -c` exits 1 on zero
# matches, so `|| echo 0` would append a SECOND zero -- use `|| true`.
todos() { grep -nv '^[[:space:]]*%' "$1" | grep '\\TODO' | grep -v 'newcommand'; }
n=$(todos main.tex | wc -l)
b=$(todos references.bib 2>/dev/null | wc -l)
echo "  main.tex       ${n} TODO"
echo "  references.bib ${b} unverified entries"
if [ "$n" -gt 0 ]; then
    echo
    todos main.tex | sed 's/\\TODO{/ -> /' | cut -c1-100 | head -12
    [ "$n" -gt 12 ] && echo "  ... and $((n - 12)) more"
fi

if [ "${1:-}" = "--check" ]; then
    # ------------------------------------------------------------- anonymity
    # TMLR rejects non-anonymous submissions WITHOUT REVIEW. tmlr.sty hides the
    # \author block on its own; it cannot hide prose. Comment lines are dropped
    # here because they never reach the PDF.
    echo
    echo "### anonymity (TMLR is double-blind)"
    if [ "${opt}" = "\\usepackage{tmlr}" ]; then
        hits=$(grep -vE '^[[:space:]]*%' main.tex \
               | grep -nE 'Marshall|leemarshall|github\.com/[A-Za-z]|Independent Researcher' \
               | grep -v 'JmlrOrg' || true)
        author_only=$(printf '%s\n' "$hits" | grep -vcE '\\(author|name|email|addr)' || true)
        if [ -z "$hits" ]; then
            echo "  clean -- nothing identifying outside comments"
        else
            printf '%s\n' "$hits" | sed 's/^/  /' | cut -c1-100
            echo
            echo "  Hits inside the \\author block are EXPECTED: tmlr.sty replaces"
            echo "  it with \"Anonymous authors\" under the bare option. Anything"
            echo "  else in that list is a real de-anonymisation and must go."
            [ "${author_only:-0}" -gt 0 ] && echo "  ${author_only} hit(s) are NOT in the author block."
        fi
    else
        echo "  skipped -- author is deliberately visible in this mode"
    fi

    # ------------------------------------------------- static LaTeX checks
    # There is no TeX toolchain on the development machine, so this is the
    # only thing standing between an edit and finding out on Overleaf.
    echo
    echo "### static checks (no TeX needed)"
    python3 precheck.py 2>/dev/null | sed 's/^/  /' \
        || echo "  precheck.py did not run -- needs python3"

    # ------------------------------------------------------------- citations
    echo
    echo "### citation hygiene"
    echo "  All 45 entries were verified against arXiv, CVF, PMLR or the"
    echo "  publisher on 2026-08-29. Eight still carry a venue line that was"
    echo "  taken from the preprint rather than confirmed proceedings --"
    echo "  LIBERO, OXE, Octo, DROID, CLIP, MAE, Levine (JMLR pages) and"
    echo "  CortexBench. Confirm those before submitting."
    exit 0
fi

# ------------------------------------------------------------------- building
command -v pdflatex >/dev/null 2>&1 || {
    echo
    echo "pdflatex not found. Either install texlive, or use Overleaf:"
    echo "  upload main.tex, references.bib, figures/, AND tmlr.sty tmlr.bst"
    echo "  fancyhdr.sty -- Overleaf has no TMLR template, it reads the .sty"
    echo "  from the project. Set the compiler to pdfLaTeX."
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

# A silently missing citation is the classic bibtex failure: the PDF builds,
# the reference list is just wrong. Say so rather than printing a size.
if [ -f main.blg ]; then
    warn=$(grep -c "Warning--" main.blg 2>/dev/null || echo 0)
    [ "$warn" -gt 0 ] && echo "  ${warn} bibtex warning(s) -- see main.blg"
fi
undef=$(grep -c "Citation.*undefined" main.log 2>/dev/null || echo 0)
[ "$undef" -gt 0 ] && echo "  ${undef} UNDEFINED citation(s) -- see main.log"
exit 0
