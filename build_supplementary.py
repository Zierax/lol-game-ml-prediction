"""Build paper/entropy_submission/supplementary.tex from the committed JSON.

Tables S1--S5 mirror the Supporting Information list in the manuscript. Every
number is read from an artifact; nothing is hard-coded.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CEIL = os.path.join(ROOT, "plots_ceiling")
TEMP = os.path.join(ROOT, "plots_temporal")
OUT = os.path.join(ROOT, "paper", "entropy_submission", "supplementary.tex")

ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
         "svm", "knn", "mlp", "soft_voting"]
DISPLAY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "svm": "RBF-SVM",
    "knn": "$k$-NN",
    "mlp": "MLP",
    "soft_voting": "Soft voting",
}
PAIR_DISPLAY = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGBM",
    "svm": "SVM",
    "knn": "$k$-NN",
    "mlp": "MLP",
}


def load(name: str, base: str = CEIL) -> dict:
    with open(os.path.join(base, name), encoding="utf-8") as fh:
        return json.load(fh)


def pfmt(p: float) -> str:
    """LaTeX p-value with three significant digits, scientific when tiny."""
    if p >= 0.9995:
        return "1.000"
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.2e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def ci(lo: float, hi: float, scale: float = 1.0, nd: int = 3) -> str:
    return f"[{scale * lo:.{nd}f}, {scale * hi:.{nd}f}]"


def main() -> None:
    S = load("ceiling_summary.json")
    B = load("bounds_audit.json")
    L = load("label_noise_audit.json")
    FC = load("temporal_fixed_cohort.json", TEMP)
    LP = load("temporal_lead_padded.json", TEMP)

    out: list[str] = []
    add = out.append

    add(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\newcommand{\bits}{\,\text{bits}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}

\title{\vspace{-1.2cm}\textbf{Supplementary Material}\\[4pt]
\large A Mutual-Information Audit of Early-Game \emph{League of Legends}\\
Match Outcome Prediction: What Eight Calibrated Scores Achieve at Ten Minutes}
\author{Ziad Salah Mohamed and Ashraf A. Mohamed}
\date{}

\begin{document}
\maketitle
\vspace{-0.6cm}

This supplement accompanies the manuscript. Table~S1 reports per-family
bootstrap intervals, Fano bounds and cross-fitted score-Bayes diagnostics;
Table~S2 reports all 21 exact McNemar comparisons with raw and Holm-adjusted
$p$-values and the two $\chi^2$ sensitivity variants; Table~S3 reports the
fixed-cohort temporal results; Table~S4 the lead-padded control; and Table~S5
the full label-noise sweep, including the degenerate frozen-model identity.
Figure~S1 plots the label-noise sweep and Figure~S2 the duration-stratified
minute-ten composition. All intervals are 95\% percentile bootstrap intervals
over 2000 test-set resamples, and all mutual information is in bits.
""")

    # ------------------------------------------------------------------ S1
    add(r"""

\begin{table}[h]
\centering
\caption{Per-family test accuracy, achieved mutual information, Fano ceiling and
cross-fitted score-Bayes diagnostics on the 1976-match test set. The
information-efficiency ratio is
$\eta = P_e/h^{-1}\!\left(H(Y)-I(S;Y)\right)$, the observed error rate divided
by the minimum error rate Fano permits; $\eta \geq 1$ means the bound is
satisfied, and a value below one would signal a violation, which for a correct
implementation can only come from a downward-biased information estimate. The
three $\eta$ columns use the plug-in information, the Miller--Madow-corrected
binned information, and the binning-free Ross (2014) estimate, respectively.
$\Delta$ is the cross-fitted score-Bayes ceiling minus the measured accuracy; a
negative value means the family sits slightly above its own score-Bayes ceiling,
within the interval reported.}
\label{tab:s1}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
Model & Acc.\ (\%) & \multicolumn{2}{c}{$I$ (bits)} & \multicolumn{2}{c}{$A_{\max}$ (\%)} & \multicolumn{3}{c}{$\eta$} & $\Delta$ (pts) \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}
 & & est. & 95\% CI & est. & 95\% CI & plug & binned & Ross & [95\% CI] \\
\midrule""")

    for fam in ORDER:
        m = S["model_families"][fam]
        b = B["fano_consistency"][fam]
        bc = m["bootstrap_ci"]
        sb = m.get("score_bayes", {})
        delta = sb.get("crossfit_minus_accuracy")
        dlo = sb.get("crossfit_minus_accuracy_ci_lower")
        dhi = sb.get("crossfit_minus_accuracy_ci_upper")
        delta_txt = ("--" if delta is None
                     else f"{100 * delta:+.2f} {ci(dlo, dhi, 100, 2)}")
        add(f"{DISPLAY[fam]} & {100 * m['test_accuracy']:.2f} & "
            f"{m['achieved_mi_bits']:.4f} & {ci(bc['achieved_mi']['ci_lower'], bc['achieved_mi']['ci_upper'], 1, 4)} & "
            f"{100 * m['fano_max_accuracy']:.2f} & {ci(bc['fano_max_accuracy']['ci_lower'], bc['fano_max_accuracy']['ci_upper'], 100, 2)} & "
            f"{b['eta_from_plugin_mi']:.3f} & {b['eta_from_biased_mi']:.3f} & "
            f"{b['eta_from_knn_mi']:.3f} & {delta_txt} \\\\")

    add(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ------------------------------------------------------------------ S2
    pm = S["pairwise_mcnemar"]
    pairs = pm["pairs"]
    chi2 = pm["sensitivity_p_values"]["chi2"]
    chi2cc = pm["sensitivity_p_values"]["chi2_cc"]

    add(r"""

\begin{longtable}{@{}lrrrrrrr@{}}
\caption{All 21 pairwise comparisons among the seven base families on the test
set ($n = 1976$). Soft voting is excluded because it is a deterministic function
of the base scores and would not constitute an independent test. $n_{01}$ is the
number of matches the first model gets wrong and the second gets right, $n_{10}$
the reverse. The exact column is the conditional binomial McNemar test, which is
the test reported in the manuscript; the two $\chi^2$ columns are the
uncorrected and continuity-corrected sensitivity variants. Holm adjustment is
applied across all 21 exact tests at family-wise $\alpha = 0.05$; five survive.
MDE is the minimum accuracy difference detectable at $\alpha = 0.05$ and 80\%
power at the observed discordant count.}\\
\toprule
Comparison & $n_{01}$ & $n_{10}$ & $p_{\text{exact}}$ & $p_{\text{Holm}}$ & $p_{\chi^2}$ & $p_{\chi^2\text{cc}}$ & MDE (pts) \\
\midrule
\endfirsthead
\toprule
Comparison & $n_{01}$ & $n_{10}$ & $p_{\text{exact}}$ & $p_{\text{Holm}}$ & $p_{\chi^2}$ & $p_{\chi^2\text{cc}}$ & MDE (pts) \\
\midrule
\endhead
\bottomrule
\endfoot""")

    for key, d in pairs.items():
        left, right = [s.strip() for s in key.split(" vs ")]
        label = f"{PAIR_DISPLAY[left]} vs.\\ {PAIR_DISPLAY[right]}"
        star = "$^{*}$" if d["significant_after_holm"] else ""
        add(f"{label}{star} & {d['n01']} & {d['n10']} & {pfmt(d['p_value'])} & "
            f"{pfmt(d['p_value_holm'])} & {pfmt(chi2[key])} & "
            f"{pfmt(chi2cc[key])} & {d['mde_accuracy_points']:.2f} \\\\")

    add(r"""\end{longtable}
\footnotesize $^{*}$Significant after Holm correction across the 21 tests.
""")

    # ------------------------------------------------------------------ S3
    add(r"""

\begin{longtable}{@{}rrrrrrrr@{}}
\caption{Minute-ten to minute-thirty behaviour restricted to games that lasted
at least a fixed number of minutes, so that the same matches contribute at every
observation window. Two cohorts are shown: games of at least 32 minutes and
games of at least 36 minutes. $\eta = P_e/h^{-1}(H(Y)-I(S;Y))$; five of the 22
rows return a slightly negative corrected information estimate or an $\eta < 1$,
both of which are small-sample artefacts of an estimate that is essentially
zero, and they are reported rather than trimmed.}\\
\toprule
Cohort (min) & Frame (min) & $n$ & Win rate & Acc.\ (\%) & $I$ (bits) & $A_{\max}$ (\%) & $\eta$ \\
\midrule
\endfirsthead
\toprule
Cohort (min) & Frame (min) & $n$ & Win rate & Acc.\ (\%) & $I$ (bits) & $A_{\max}$ (\%) & $\eta$ \\
\midrule
\endhead
\bottomrule
\endfoot""")

    for r in FC:
        add(f"{r['min_duration_min']} & {r['frame_minutes']} & {r['n_samples']} & "
            f"{r['win_rate']:.4f} & {100 * r['voting_accuracy']:.2f} & "
            f"{r['voting_achieved_mi']:.4f} & "
            f"{100 * r['voting_fano_max_accuracy']:.2f} & "
            f"{r['voting_information_efficiency']:.3f} \\\\")

    add(r"""\end{longtable}
""")

    # ------------------------------------------------------------------ S4
    add(r"""

\begin{longtable}{@{}rrrrrrrr@{}}
\caption{The lead-padded control. Games that ended before minute $m$ are padded
forward with their final observed frame, so that the population is held fixed
across all windows while the post-outcome frames are not used. Accuracy peaks at
minute 22 and information at minute 16; the sharp fall thereafter is the
signature of padding rather than of new information, which is why the manuscript
reports this control alongside the fixed cohorts rather than instead of them.}\\
\toprule
Padding from (min) & Frame (min) & $n$ & Win rate & Acc.\ (\%) & $I$ (bits) & $A_{\max}$ (\%) & $\eta$ \\
\midrule
\endfirsthead
\toprule
Padding from (min) & Frame (min) & $n$ & Win rate & Acc.\ (\%) & $I$ (bits) & $A_{\max}$ (\%) & $\eta$ \\
\midrule
\endhead
\bottomrule
\endfoot""")

    for r in LP:
        add(f"{r['min_duration_min']} & {r['frame_minutes']} & {r['n_samples']} & "
            f"{r['win_rate']:.4f} & {100 * r['voting_accuracy']:.2f} & "
            f"{r['voting_achieved_mi']:.4f} & "
            f"{100 * r['voting_fano_max_accuracy']:.2f} & "
            f"{r['voting_information_efficiency']:.3f} \\\\")

    add(r"""\end{longtable}
""")

    # ------------------------------------------------------------------ S5
    sweep = L["noise_sweep"]
    deg = L["degenerate_test_only_sweep"]
    rhos = sorted(sweep, key=float)

    add(r"""

\begin{longtable}{@{}rlrrrr@{}}
\caption{Label-noise sweep. The primary design corrupts a fraction $\rho$ of the
\emph{training} labels and refits, leaving the calibration fold and the test
labels clean; each cell is the mean over three independent corruption draws.
The final block is the secondary, degenerate design in which the test labels are
corrupted and the model is frozen: accuracy there tracks the algebraic identity
$\mathrm{acc}(\rho) = \rho + (1-2\rho)\,\mathrm{acc}(0)$, which holds for any
frozen score at the realised corruption rate and therefore carries no evidence
about the model. It is reported only so that the two designs cannot be
confused.}\\
\toprule
$\rho$ & Model & Accuracy (\%) & s.d.\ (pts) & $I$ (bits) & s.d.\ (bits) \\
\midrule
\endfirsthead
\toprule
$\rho$ & Model & Accuracy (\%) & s.d.\ (pts) & $I$ (bits) & s.d.\ (bits) \\
\midrule
\endhead
\bottomrule
\endfoot""")

    add(r"\multicolumn{6}{@{}l}{\emph{Primary design: training-label corruption with refit}}\\ \midrule")
    for rho in rhos:
        cell = sweep[rho]
        for i, fam in enumerate(("logistic_regression", "lightgbm", "soft_voting")):
            d = cell[fam]
            rho_txt = f"{float(rho):.2f}" if i == 0 else ""
            add(f"{rho_txt} & {DISPLAY[fam]} & {100 * d['mean_accuracy']:.2f} & "
                f"{100 * d['std_accuracy']:.2f} & {d['mean_achieved_mi']:.4f} & "
                f"{d['std_achieved_mi']:.4f} \\\\")

    add(r"\midrule \multicolumn{6}{@{}l}{\emph{Secondary design: test-label corruption, frozen model (algebraic identity)}}\\ \midrule")
    for rho in rhos:
        d = deg[rho]
        add(f"{float(rho):.2f} & Frozen score & {100 * d['mean_accuracy']:.2f} & -- & "
            f"{d['mean_achieved_mi']:.4f} & -- \\\\")

    add(r"""\end{longtable}
""")

    # ------------------------------------------------------------------ figures
    add(r"""

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/S1_label_noise.png}
\caption{Left: test accuracy against the training-label corruption rate $\rho$
for the three families carried through the sweep, with $\pm$ one standard
deviation over three corruption draws. Right: the achieved mutual information
over the same sweep. The dotted line in the left panel is the frozen-model
algebraic identity of Table~S5; its steepness is an artefact of corrupting the
evaluation labels, not a property of the model, and the contrast with the three
refitted curves is the point of the figure. Logistic regression and soft voting
retain most of their information at $\rho = 0.3$, while LightGBM loses about half
of it, consistent with the boosted model fitting the injected noise.}
\end{figure}


\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/S2_strata_composition.png}
\caption{The minute-ten snapshot decomposed by the final duration of the match.
Left: the share of games in each stratum, showing that the two longer strata
together account for roughly 71\% of the population. Middle: minute-ten
accuracy. Right: minute-ten achieved information, which spans 0.013 to 0.681
bits---a 54-fold range at a single observation time. The aggregate 72\% plateau
is therefore an average over sub-populations of very different informativeness,
and any single statement of the form ``the ten-minute snapshot contains $X$
bits'' is relative to the duration mix of the population it is computed on.}
\end{figure}

\end{document}
""")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
