"""Static checks on main.tex that do not need a TeX installation.

Not a substitute for compiling. Catches the failures that are actually common:
unbalanced braces, cite keys with no bib entry, bib entries never cited, and
package loads that clash with what tmlr.sty already pulls in.
"""
import io
import re
import sys

tex = io.open("main.tex", encoding="utf-8").read()
bib = io.open("references.bib", encoding="utf-8").read()

# strip full-line comments; TeX comments cannot cause build errors
body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))

fail = 0

# --- 1. brace balance -----------------------------------------------------
depth = 0
bad_line = None
for i, line in enumerate(body.splitlines(), 1):
    prev = ""
    for ch in line:
        if prev != "\\":
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0 and bad_line is None:
                    bad_line = i
        prev = "" if prev == "\\" else ch
print("1. brace balance      : depth %+d %s" % (depth, "OK" if depth == 0 else "MISMATCH"))
if depth != 0:
    fail += 1

# --- 2. cite keys vs bib entries -----------------------------------------
cited = set()
for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", body):
    cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
defined = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

missing = sorted(cited - defined)
unused = sorted(defined - cited)
print("2. citations          : %d cited, %d defined" % (len(cited), len(defined)))
if missing:
    print("   UNDEFINED (build would print 'Citation undefined'):")
    for k in missing:
        print("     -", k)
    fail += 1
else:
    print("   every \\cite key resolves")
if unused:
    print("   %d entries never cited (harmless, bibtex drops them):" % len(unused))
    print("     " + ", ".join(unused[:8]) + (" ..." if len(unused) > 8 else ""))

# --- 3. package clashes with tmlr.sty ------------------------------------
try:
    sty = io.open("tmlr.sty", encoding="utf-8").read()
except OSError:
    print("3. package clashes    : tmlr.sty NOT FOUND")
    sys.exit(1)

sty_pkgs = set(re.findall(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]*)\}", sty))
sty_pkgs = {p.strip() for grp in sty_pkgs for p in grp.split(",")}
tex_loads = re.findall(r"\\usepackage(\[[^\]]*\])?\{([^}]*)\}", body)

clash = []
for opts, grp in tex_loads:
    for p in (x.strip() for x in grp.split(",")):
        if p in sty_pkgs and p != "tmlr":
            clash.append((p, opts or "no options"))
print("3. package clashes    : tmlr.sty loads {%s}" % ", ".join(sorted(sty_pkgs)))
if clash:
    for p, o in clash:
        sev = "OPTION CLASH -- build stops" if o != "no options" else "duplicate, harmless"
        print("   %-12s reloaded with %-14s %s" % (p, o, sev))
        if o != "no options":
            fail += 1
else:
    print("   no package is loaded twice")

# --- 4. macro contract in both directions --------------------------------
# main.tex USES \name \email \addr, which tmlr.sty must define.
# tmlr.sty USES \openreview \month \year, which main.tex must define -- the
# accepted branch expands them and an undefined one is "Undefined control
# sequence" at camera-ready, i.e. exactly when there is no time to fix it.
print("4. macro contract")
absent = [m for m in ("name", "email", "addr")
          if ("\\" + m) in body and not re.search(r"\\def\\" + m + r"\b", sty)]
print("   main.tex needs from tmlr.sty :",
      "name, email, addr all defined" if not absent else "MISSING " + ", ".join(absent))
fail += bool(absent)

owed = [m for m in ("openreview", "month", "year")
        if re.search(r"\\" + m + r"\b", sty) and not re.search(r"\\def\\" + m + r"\b", body)]
print("   tmlr.sty needs from main.tex :",
      "openreview, month, year all defined" if not owed else "MISSING " + ", ".join(owed))
fail += bool(owed)

print()
print("FAILURES:", fail)
sys.exit(1 if fail else 0)
