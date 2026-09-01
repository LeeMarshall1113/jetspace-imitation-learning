# Vendored style files

`tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` are not this project's work. They are
the official TMLR style package, copied here unmodified so the paper builds
from a clean checkout with no network and cannot silently change under it if
upstream is edited.

| file | sha256 (first 16) |
|---|---|
| `tmlr.sty` | `816214ff5919aa45` |
| `tmlr.bst` | `306fd454cf40771b` |
| `fancyhdr.sty` | `3d2922548e0e5f1a` |

- **Source** — <https://github.com/JmlrOrg/tmlr-style-file>
- **Pinned at** — commit `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`, fetched 2026-08-31
- **Licence** — Apache 2.0, the same licence as this repository, so no
  additional obligation beyond attribution. This file is that attribution.

To refresh them, re-fetch from the URL above and update the hashes here. Do not
edit them in place: TMLR checks the formatting against its own style, and a
local modification is the kind of thing that is invisible until a desk reject.

`main.pdf` is gitignored, so nothing built from these files is committed.
