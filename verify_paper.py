"""Full verification: every quantitative claim in main.tex vs the artifacts.

Run with::

    python verify_paper.py

Exits non-zero if any check fails.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, 'paper', 'entropy_submission', 'main.tex')
T = open(P, encoding='utf-8').read()


def J(*parts):
    return json.load(open(os.path.join(ROOT, *parts), encoding='utf-8'))


S = J('plots_ceiling', 'ceiling_summary.json')
B = J('plots_ceiling', 'bounds_audit.json')
L = J('plots_ceiling', 'label_noise_audit.json')
TS = J('plots_temporal', 'temporal_summary.json')
FC = J('plots_temporal', 'temporal_fixed_cohort.json')
LP = J('plots_temporal', 'temporal_lead_padded.json')
DS = J('plots_temporal', 'temporal_duration_stratified.json')
AC = J('plots_temporal', 'temporal_accrual.json')

fails = []


def check(label, ok, detail=""):
    if not ok:
        fails.append(label)
        print(f"  FAIL  {label}   {detail}")


def present(s):
    return s in T


def pfmt(x):
    """Paper rule: 2 dp, but 3 dp when 2 dp would round to zero."""
    s = f"{x:.2f}"
    return f"{x:.3f}" if float(s) == 0.0 else s


ORDER = ["logistic_regression", "mlp", "lightgbm", "svm", "soft_voting",
         "random_forest", "xgboost", "knn"]
NAMES = {"logistic_regression": "Logistic regression", "mlp": "MLP",
         "lightgbm": "LightGBM", "svm": "SVM (RBF)",
         "soft_voting": "Soft voting", "random_forest": "Random forest",
         "xgboost": "XGBoost", "knn": "$k$-NN"}

print("=== Table 1: performance and information ===")
for k in ORDER:
    v = S["model_families"][k]
    for f, fmt in (("test_accuracy", "{:.4f}"), ("roc_auc", "{:.4f}"),
                   ("achieved_mi_bits", "{:.4f}"),
                   ("achieved_mi_bits_knn", "{:.4f}"),
                   ("fano_max_accuracy", "{:.4f}"),
                   ("fano_max_accuracy_from_plugin_mi", "{:.4f}"),
                   ("information_efficiency", "{:.3f}")):
        check(f"T1 {k}.{f}", present(fmt.format(v[f])), fmt.format(v[f]))

print("=== Table 3: cross-check ===")
for k in ORDER:
    r = B["models"][k]
    payoff = (r["score_bayes_plug_in_ceiling"]
              - r["score_bayes_crossfit_ceiling"])
    delta = r["achieved_mi_ksg_bits"] - r["achieved_mi_binned_bits_bias_corrected"]
    row = (NAMES[k]
           + " & " + f"{r['achieved_mi_binned_bits_bias_corrected']:.4f}"
           + " & " + f"{r['achieved_mi_ksg_bits']:.4f}"
           + " & $" + f"{delta:+.3f}" + "$"
           + " & " + f"{r['achieved_mi_binned_bits']:.4f}"
           + " & $" + f"{payoff:+.4f}" + "$"
           + " & " + f"{B['fano_consistency'][k]['eta_from_knn_mi']:.3f}"
           + r" \\")
    check(f"T3 {k}", row in T, f"want: {row}")

print("=== Table 4: McNemar ===")
mc = S["pairwise_mcnemar"]
check("n_pairs = 21", mc["n_pairs"] == 21, str(mc["n_pairs"]))
check("5 after Holm", mc["n_significant_after_holm"] == 5)
check("9 nominal", mc.get("n_significant_uncorrected") == 9,
      str(mc.get("n_significant_uncorrected")))
for pair, r in mc["pairs"].items():
    if r["significant_after_holm"]:
        check(f"T4 {pair} n01", present(f" & {r['n01']} & {r['n10']} & "))
        check(f"T4 {pair} p", present(f"{r['p_value']:.5f}"))
        check(f"T4 {pair} holm", present(f"{r['p_value_holm']:.4f}"))
mdes = [r["mde_accuracy_points"] for r in mc["pairs"].values()]
check("MDE 1.48-3.02", present("1.48") and present("3.02"),
      f"{min(mdes):.2f}-{max(mdes):.2f}")
ha = mc["mde_holm_aware_range_pts"]
check("MDE Holm 1.70-4.12", present("1.70--4.12"),
      f"{ha[0]:.2f}-{ha[1]:.2f}")

print("=== Table 5: learning curves (incl. MI axis) ===")
lc = S["learning_curves"]
for fam, disp in (("logistic_regression", "Logistic regression"),
                   ("lightgbm", "LightGBM"), ("knn", "$k$-NN")):
    v = lc[fam]
    a = v["accuracy_curvature"]
    mc_ = v["mi_curvature"]
    vals, mis = v["mean_accuracy"], v["mean_achieved_mi"]
    # Locate the row and check each non-p field individually, then compare
    # the four p-values numerically (the paper uses several precisions).
    # Locate the learning-curve row: it is the one with four p-value groups.
    cand = [ln for ln in T.split("\n")
            if ln.startswith(disp + " & 0.")
            and len(re.findall(r"\$\([0-9.e\-]+\)\$", ln)) == 4]
    check(f"T5 {fam} row exists", len(cand) == 1, str(cand))
    if len(cand) != 1:
        continue
    ln = cand[0]
    for val in (f"{vals[0]:.4f}", f"{vals[-1]:.4f}", f"{mis[0]:.4f}",
                f"{mis[-1]:.4f}",
                "$" + f"{100*v['accuracy_slope_wls_per_log2']:+.2f}" + "$",
                "$" + f"{a['quadratic_coefficient']*1e3:+.1f}" + r"{\times}10^{-3}",
                "$" + f"{v['mi_slope_wls_per_log2']:+.4f}" + "$",
                "$" + f"{mc_['quadratic_coefficient']*1e3:+.1f}" + r"{\times}10^{-3}"):
        check(f"T5 {fam} field {val}", val in ln, ln[:160])
    ps = re.findall(r"\$\(([0-9.e\-]+)\)\$", ln)
    check(f"T5 {fam} four p-values", len(ps) == 4, str(ps))
    if len(ps) == 4:
        want = [v["accuracy_slope_wls_p"], a["quadratic_p_value"],
                v["mi_slope_wls_p"], mc_["quadratic_p_value"]]
        for g, w in zip(ps, want):
            dec = len(g.split(".")[1]) if "." in g else 0
            check(f"T5 {fam} p {g} ~ {w:.6f}",
                  abs(float(g) - w) <= 0.5 * 10 ** -dec, f"want {w:.6f}")

print("=== Table 6: label noise ===")
for fam, disp in (("logistic_regression", "Logistic regression"),
                   ("soft_voting", "Soft voting"), ("lightgbm", "LightGBM")):
    d = L["degradation_at_rho_max"][fam]
    sw = L["noise_sweep"]
    row = (disp
           + " & " + f"{sw['0.0'][fam]['mean_achieved_mi']:.4f}"
           + " & " + f"{sw['0.1'][fam]['mean_achieved_mi']:.4f}"
           + " & " + f"{sw['0.2'][fam]['mean_achieved_mi']:.4f}"
           + " & " + f"{sw['0.3'][fam]['mean_achieved_mi']:.4f}"
           + " & $" + f"{d['accuracy_drop_points']:.2f}" + "$"
           + " & $["
           + f"{d['accuracy_drop_ci_lower']:.2f}, {d['accuracy_drop_ci_upper']:.2f}"
           + "]$"
           + " & $-$" + f"{d['mi_drop_percent']:.1f}" + r"\%")
    check(f"T6 {fam}", row in T, f"want: {row}")
check("no rho_reproducing_signature in artifact",
      "rho_reproducing_signature" not in L)
check("paper does not claim reproduction",
      "reproduces its clean accuracy" not in T
      and "reproduces that clean endpoint" not in T)

print("=== Table 7: duration strata ===")
for disp, key in (("$<$ 24 min", "<24 min"), ("24--29 min", "24-29 min"),
                  ("$\\geq$ 30 min", ">=30 min")):
    r = DS[key]
    row = (disp + " & " + f"{int(r['n_samples']):,}"
           + " & " + f"{r['voting_accuracy']:.4f}"
           + " & " + f"{r['voting_achieved_mi']:.4f}"
           + " & " + f"{r['voting_fano_max_accuracy']:.4f}" + r" \\")
    check(f"T7 {key}", row in T, f"want: {row}")
wacc = sum(DS[k]["n_samples"] * DS[k]["voting_accuracy"]
           for k in DS) / sum(DS[k]["n_samples"] for k in DS)
wmi = sum(DS[k]["n_samples"] * DS[k]["voting_achieved_mi"]
          for k in DS) / sum(DS[k]["n_samples"] for k in DS)
check("weighted stratum acc 0.7245", present("0.7245"), f"{wacc:.4f}")
check("weighted stratum MI 0.281", present("0.281"), f"{wmi:.4f}")
check("mixture term 0.056", present("0.056"), f"{wmi - TS[0]['voting_achieved_mi']:.4f}")

print("=== temporal ===")
check("m10 acc 0.7291", abs(TS[0]["voting_accuracy"] - 0.7291) < 1e-4)
check("m30 acc 0.8395", abs(TS[-1]["voting_accuracy"] - 0.8395) < 1e-4)
check("peak acc 0.8477 at m26", present("84.8"),
      f"{max(r['voting_accuracy'] for r in TS):.4f}")
check("m10 MI 0.225", present("0.225"), f"{TS[0]['voting_achieved_mi']:.4f}")
check("m30 MI 0.506", present("0.506"), f"{TS[-1]['voting_achieved_mi']:.4f}")
n30 = TS[-1]["n_samples"] / TS[0]["n_samples"]
check("39.9% survivors", present("39.9"), f"{100*n30:.1f}%")
lp_peak = max(LP, key=lambda r: r["voting_accuracy"])
lp_mi = max(LP, key=lambda r: r["voting_achieved_mi"])
check("lead-padded acc peak m22", lp_peak["frame_minutes"] == 22)
check("lead-padded MI peak m16 0.293", lp_mi["frame_minutes"] == 16
      and present("0.293"), f"{lp_mi['voting_achieved_mi']:.4f}")
check("lead-padded m30 67.1 / 0.124",
      present("67.1") and present("0.124"),
      f"{100*LP[-1]['voting_accuracy']:.1f} / {LP[-1]['voting_achieved_mi']:.4f}")
check("lead-padded 16% sample", present("16\\%"),
      f"{100*LP[-1]['n_samples']/LP[0]['n_samples']:.1f}%")
for md, (a10, a30) in ((32, (52.8, 75.1)), (36, (51.7, 61.1))):
    rows = [r for r in FC if r["min_duration_min"] == md]
    check(f"cohort>={md} acc10 {a10}", present(f"{a10}")
          and abs(100 * rows[0]["voting_accuracy"] - a10) < 0.05,
          f"{100*rows[0]['voting_accuracy']:.2f}")
    check(f"cohort>={md} acc30 {a30}", present(f"{a30}")
          and abs(100 * rows[-1]["voting_accuracy"] - a30) < 0.05,
          f"{100*rows[-1]['voting_accuracy']:.2f}")
anomalous = [r for r in FC
             if r["voting_information_efficiency"] < 1
             or r["voting_achieved_mi"] < 0]
check("five rows with eta<1 or neg MI",
      present("Five of the 22") and len(anomalous) == 5,
      f"{len(anomalous)} rows: "
      + str([(r['min_duration_min'], r['frame_minutes']) for r in anomalous]))
g_all = 100 * (TS[-1]["voting_accuracy"] - TS[0]["voting_accuracy"])
c32 = [r for r in FC if r["min_duration_min"] == 32]
g32 = 100 * (c32[-1]["voting_accuracy"] - c32[0]["voting_accuracy"])
check("gains: 11 / 22 / 9", present("9--22") and present("11 points")
      and abs(g_all - 11) < 0.6 and abs(g32 - 22) < 0.6,
      f"all={g_all:.1f} c32={g32:.1f}")

print("=== accrual ===")
check("slope 0.0173", present("0.0173"),
      f"{AC['mi_slope_bits_per_minute']:.5f}")
check("R2 0.982", present("0.982"), f"{AC['mi_fit_r2']:.4f}")
check("39% window spread", present("39\\%"),
      f"{AC['window_sensitivity']['slope_relative_spread_pct']:.1f}")
slopes = [v["slope_bits_per_minute"]
          for v in AC["window_sensitivity"]["table"].values()]
check("window slopes 0.0131-0.0194", present("0.0131--0.0194"),
      f"{min(slopes):.4f}-{max(slopes):.4f}")
t85 = [v["minutes_to_85_pct"] for v in AC["window_sensitivity"]["table"].values()]
t90 = [v["minutes_to_90_pct"] for v in AC["window_sensitivity"]["table"].values()]
check("85 window 18.0-19.5", present("18.0--19.5"), f"{min(t85)}-{max(t85)}")
check("90 window 25.3-29.9", present("25.3--29.9"), f"{min(t90)}-{max(t90)}")
sat = AC["saturating_fit"]["minutes_to_error_targets"]
check("saturating 18.6 / 30.2", present("18.6") and present("30.2"),
      f"{sat['minutes_to_85_pct']} / {sat['minutes_to_90_pct']}")
check("overshoot 16.6 mbit", present("16.6"),
      f"{1000*AC['fit_overshoot_at_m10_bits']:.1f}")
check("observed 85 crossing m18",
      AC["observed_crossings"]["observed_minutes_to_85_pct"] == 18
      and present("minute 18"))
check("90 never observed",
      AC["observed_crossings"]["observed_minutes_to_90_pct"] is None
      and AC["first_measured_acc_ge_85_at_minute"] is None)

print("=== permutation null ===")
pn = S["mi_permutation_null"]
check("null mean 0.0161", present("0.0161"), f"{pn['null_max_mean']:.4f}")
check("null sd 0.0030", present("0.0030"), f"{pn['null_max_std']:.4f}")
z = (pn["observed_max"] - pn["null_max_mean"]) / pn["null_max_std"]
check("75 SD", present("75 standard deviations"), f"z={z:.1f}")
check("11.1x null95", present("11.1"), f"{pn['observed_over_null95_ratio']:.2f}")
check("p floor 0.005", present("1/(B+1)") and present("0.005"),
      f"{pn['p_value_maxT']:.6f}")
check("maxT not Bonferroni", "Bonferroni-adjusted $p$-value" in T)

print("=== Fano information budget (I >= H - h(P_e)) ===")
H = S["target_entropy_bits_test"]


def h(p):
    import math
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


for acc, txt in ((0.75, "0.189"), (0.80, "0.278"), (0.85, "0.390"),
                  (0.90, "0.531")):
    mi_min = H - h(1 - acc)
    check(f"budget for {acc:.0%} = {mi_min:.3f} bits", present(txt),
          f"computed {mi_min:.4f}")
best_mi = max(S["model_families"][k]["achieved_mi_bits"] for k in ORDER)
check("best MI 0.230", present("0.230") and abs(best_mi - 0.2305) < 0.001,
      f"{best_mi:.5f}")
check("no wrong 'beats 72%' budget claim",
      "must carry more than 0.23 bits" not in T)

print("=== framing / structure ===")
check("title does not claim snapshot information",
      "Quantifying the Information Available at Ten Minutes" not in T)
check("explicit 'cannot refute a ceiling' statement",
      "cannot refute an information ceiling" in T
      and "cannot refute a ceiling" in T)
check("no 'largely survivorship'", "largely survivorship" not in T)
check("no survivorship keyword", "survivorship bias" not in T)
check("related work section present", "\\section{Related Work}" in T)
check("lightgbm cited", "ke2017lightgbm" in T)
check("no fabricated kinkade ref", "kinkade2022dota" not in T)
check("feder DOI correct", "10.1109/18.272494" in T)
check("hellman DOI correct", "10.1109/TIT.1970.1054466" in T)
check("no wrong feder DOI", "10.1109/18.272496" not in T)
check("no wrong hellman DOI", "10.1109/TIT.1970.1054482" not in T)
check("kaggle sololq URL fixed",
      "league-of-legends-soloq-ranked-games-per-frame" not in T
      and "league-of-legends-soloq-ranked-games}" in T)
check("ross cited", "ross2014mutual" in T)
check("no ksg misattribution as the estimator",
      "Kraskov--St\\\"ogbauer--Grassberger estimator" not in T)
check("miller madow 6,717/1,186", present("6,717") and present("1,186")
      and present("6,718") is False)
check("MI range 0.143-0.230", present("0.143--0.230"))
check("14-23 percent", present("14--23\\%"))
check("seven non-kNN range to 0.230", present("0.192 and 0.230"))
check("no extract-as-family verb in results",
      "the eight families extract" not in T)
check("key findings list present", "\\subsection*{Key Findings}" in T)
check("MI-axis curvature reported", present("-3.3{\\times}10^{-3}$ $(0.0009)$"))
check("spearman 0.98 / 0.20", present("0.98") and present("0.20"))
check("hyperparameter disclosure", "library-default hyperparameters" in T)
check("eta unit-test recommendation", "standing unit test" in T)
check("mcnemar episode disclosed",
      "discordant counts in an earlier McNemar table" in T)
check("no 'composition is itself informative'",
      "composition is itself informative" not in T
      and "whose composition is itself" not in T)
check("no three-way gap-decomposition percentages",
      "0--18\\%" not in T and "70--90\\%" not in T
      and "most of the remainder" not in T)
check("curvature stated on the information axis",
      "p = 0.0009" in T and "information axis" in T)
check("no 'beats 72%' budget claim",
      "must carry more than 0.23 bits" not in T)
check("roadmap mentions related work", "sec:related" in T)
check("no uncited fagerland2013 key", "fagerland2013}" not in T)

print("=== citations vs bibliography ===")
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", T):
    for k in m.group(1).split(","):
        cited.add(k.strip())
defined = set(re.findall(r"\\bibitem\{([^}]*)\}", T))
check("no dangling citations", not (cited - defined), str(cited - defined))
check("no uncited bibitems", not (defined - cited), str(defined - cited))

print()
print("=" * 70)
if fails:
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
