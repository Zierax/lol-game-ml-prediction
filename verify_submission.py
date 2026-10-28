"""Final submission integrity check.

Confirms that the files inside the submission ZIP are byte-identical to the
files in the working tree, that the PDF is newer than the LaTeX source, and
that every artifact the paper cites is present in the package.
"""
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, 'paper', 'entropy_submission')
ZIP = os.path.join(ROOT, 'Entropy_Submission_LOL_Information_Audit.zip')

fails = []


def check(label, ok, detail=""):
    print(("  pass  " if ok else "  FAIL  ") + label + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== package contents match the working tree ===")
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    for name in names:
        norm = name.replace("\\", "/")
        src = os.path.join(PKG, norm)
        if not os.path.isfile(src):
            check(f"{norm} exists in tree", False)
            continue
        with z.open(name) as fh:
            zh = hashlib.sha256(fh.read()).hexdigest()
        check(f"{norm}", zh == sha(src), "hash mismatch")

print("\n=== required files present ===")
for rel in ["main.tex", "main.pdf", "supplementary.tex", "supplementary.pdf",
            "Definitions/mdpi.cls",
            "Definitions/mdpi.bst", "Definitions/verse.sty",
            "figures/achieved_mi.png", "figures/fano_curve.png",
            "figures/learning_curve.png", "figures/mi_spectrum.png",
            "figures/minute10_strata.png", "figures/pca_variance.png",
            "figures/temporal_ceiling.png",
            "figures/S1_label_noise.png", "figures/S2_strata_composition.png",
            "artifacts/ceiling_summary.json",
            "artifacts/bounds_audit.json",
            "artifacts/label_noise_audit.json",
            "artifacts/temporal_summary.json",
            "artifacts/temporal_fixed_cohort.json",
            "artifacts/temporal_lead_padded.json",
            "artifacts/temporal_duration_stratified.json",
            "artifacts/temporal_accrual.json",
            "artifacts/reference_check.json"]:
    with zipfile.ZipFile(ZIP) as z:
        got = any(n.replace("\\", "/") == rel for n in z.namelist())
    check(rel, got)

print("\n=== PDFs are current ===")
for stem in ("main", "supplementary"):
    tex = os.path.getmtime(os.path.join(PKG, f"{stem}.tex"))
    pdf = os.path.getmtime(os.path.join(PKG, f"{stem}.pdf"))
    check(f"{stem}.pdf newer than {stem}.tex", pdf >= tex)

print("\n=== no build junk in package ===")
with zipfile.ZipFile(ZIP) as z:
    junk = [n for n in z.namelist()
            if n.lower().endswith((".aux", ".log", ".out", ".toc", ".blg",
                                   ".bbl", ".lof", ".lot"))]
check("no aux/log/toc files", not junk, str(junk))

print("\n" + "=" * 70)
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("SUBMISSION PACKAGE VERIFIED")
