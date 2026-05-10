# Part B — Counterfactual Interpretation

## Main Figure

The required Part B figure is `figures/fig_counterfactual_main_svar.png`. It is the headline figure for the submission and shows only two series on the same axes: Ukraine actual year-on-year inflation and the main SVAR counterfactual inflation path under hypothetical Euro Area membership. The optional bootstrap confidence band is included only to communicate uncertainty around the main SVAR result.

![Main SVAR counterfactual](figures/fig_counterfactual_main_svar.png)

*Figure B1. Main SVAR counterfactual: actual Ukraine inflation, the Ukraine-in-the-Euro-Area counterfactual, and the SVAR confidence band. The shaded yellow, purple, and red windows mark the GFC, the Crimea/Donbas crisis, and the full-scale invasion.*

`figures/fig_counterfactual_robustness.png`, `figures/fig_svar_stationarity_robustness.png`, `figures/fig_robustness_stationarity_svar.png`, `figures/fig_robustness_reduced_form_projection.png`, and `figures/fig_robustness_ascm.png` are appendix-style robustness figures and should not be read as alternative headline results.

## Robustness Figures

The robustness checks are generated in two formats. First, the combined appendix plots compare the main SVAR result to the other counterfactual benchmarks. Second, each robustness method is also plotted individually against actual inflation and the main SVAR counterfactual for reference. These files are kept as appendix material rather than displayed as headline figures in Part B.

## Methodological Overview

The main specification is the **Blanchard-Quah SVAR**. Euro Area membership is operationalized as a shock-replacement exercise: Ukraine-specific supply shocks are preserved, while Ukraine domestic demand and monetary shocks are replaced with Euro Area demand shocks. Demand and monetary shocks are restricted to have zero long-run effect on output, while supply shocks may have permanent effects on output.

Part A matters directly for the construction of Part B. The regime chronology determines the treatment intensity of hypothetical Euro Area membership. The treatment is smaller during de facto peg periods, larger during the devaluation episodes and the post-2016 inflation-targeting period, and constrained again during the wartime fixed-rate and capital-control regime.

The reduced-form projection benchmark and the ASCM specification are retained only as robustness checks. The reduced-form projection is a single-horizon OLS benchmark, not a local projection. The stationarity-robustness SVAR is the main robustness exercise because it preserves the structural identification logic of the baseline model.

## Stationarity Robustness

The baseline SVAR in levels exhibits instability because Ukraine inflation contains large structural breaks and crisis episodes. As a robustness exercise, the model is re-estimated on first-differenced inflation while preserving the same Blanchard-Quah identification and shock-replacement logic. Structural IRFs for the differenced system are cumulated back to inflation levels so the counterfactual remains directly comparable to the baseline figure.

The qualitative conclusions are stable across the two SVAR specifications. The stationarity-robustness SVAR delivers larger magnitudes in the same key episodes: **+9.5 pp** in 2008-09, **+27.6 pp** in 2014-15, **-4.4 pp** after 2016, and **-8.4 pp** in 2022-23. The baseline interpretation therefore does not rest only on the unstable level VAR.

## Interpretation

The main SVAR counterfactual suggests that monetary sovereignty was inflationary during Ukraine's major devaluation crises, but that it became more valuable during the full-scale war once the exchange rate, capital controls, and emergency liquidity tools were actively used as crisis-management instruments.

**2008-09 global financial crisis.** The main SVAR gap is **+5.7 pp**: actual inflation averages **19.9%**, versus a counterfactual **14.2%**. The implication is that the hryvnia devaluation transmitted inflation that Euro Area membership would likely have muted.

**2014-15 Crimea/Donbas crisis.** This is the largest inflation cost of monetary sovereignty in the sample. The main SVAR gap is **+18.1 pp**: actual inflation averages **31.6%**, versus a counterfactual **13.5%**. The large devaluation and credibility collapse dominate the inflation outcome.

**Post-2016 inflation-targeting period.** The sign reverses. The main SVAR gap is **-2.1 pp**: actual inflation averages **9.1%**, versus a counterfactual **11.2%**. Once the NBU established a more credible inflation-targeting framework, the marginal inflation benefit of Euro Area membership appears much smaller.

**2022 full-scale invasion.** The main SVAR gap is **-6.8 pp**: actual inflation averages **20.5%**, versus a counterfactual **27.4%**. Under this interpretation, monetary sovereignty provided crisis-management tools that would have been harder to deploy inside the Euro Area.

The broad message is therefore asymmetric. Euro membership would likely have lowered inflation during the two major devaluation crises, but the post-2016 and wartime episodes show that the value of monetary sovereignty depends on the institutional regime and the type of shock hitting the economy.
