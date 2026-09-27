# Final-paper figure to retained-source mapping

The file names preserve historical source numbering; they are not silently renamed to final paper numbers. The CSV/JSON material is saved plot input or provenance, not a complete guarantee of the final submitted figure's pixel-identical appearance. Historical archive executors are not distributed in the public release.

| Final paper figure | Evidence scope | Retained repository source | Historical source number / limit |
|---|---|---|---|
| Figure 1 | Conceptual diagram | Main manuscript figure asset not distributed | No numerical reproduction claim. |
| Figure 2 | Confirmatory S3–S5 | [`results/figures/main_2/`](../results/figures/main_2/), [`results/confirmatory/`](../results/confirmatory/) | Saved Origin columns/elements and confirmatory summaries; final artwork not included. |
| Figure 3 | Confirmatory S3–S5 | [`results/figures/main_3/`](../results/figures/main_3/), [`results/figures/late_confirmatory/图3数据.csv`](../results/figures/late_confirmatory/图3数据.csv) | Saved effects/full grid and late plot data; no original per-sample scores. |
| Figure 4 | Historical S0–S2 | [`results/figures/main_4/figure3_plot_data.csv`](../results/figures/main_4/figure3_plot_data.csv) | Historical source Figure 3; saved density/statistic coordinates. |
| Figure 5 | Historical S0–S2 | [`results/figures/main_5/figure6_plot_data.csv`](../results/figures/main_5/figure6_plot_data.csv) | Historical source Figure 6; saved conditional filtering coordinates. |
| Figure 6 | Historical S0–S2 | [`results/figures/main_6/figure9_plot_data.csv`](../results/figures/main_6/figure9_plot_data.csv) | Historical source Figure 9; saved rank-change/case summaries. |
| Figure 7 | Historical S0–S2 | [`results/figures/main_7/figure10_plot_data.csv`](../results/figures/main_7/figure10_plot_data.csv) | Historical source Figure 10; image-bearing NPZ excluded. |
| Figure 8 | Historical S0–S2 | [`results/figures/main_8/Figure7_FP32数据.csv`](../results/figures/main_8/Figure7_FP32数据.csv) | Historical source Figure 7; approved FP32 beta sensitivity authority. |
| Figure 9 | Historical S0–S2 | [`results/figures/main_9/figure8_plot_data.csv`](../results/figures/main_9/figure8_plot_data.csv), [`results/figures/main_9/authority/`](../results/figures/main_9/authority/) | Historical source Figure 8; matched latency/performance and component profiling. |
| Extended appendix Figure R1 | Historical appendix | [`docs/FigS1.png`](FigS1.png) | Former Supplementary Figure S1; included scientific plot, not a dataset image. |

The `main_9` plot-input copies and `main_9/authority` source-authority files can have identical hashes while retaining different provenance roles; the same applies to `main_8` and `main_8/authority`. They are intentionally retained as separate identities. The duplicate `late_efficiency` presentation copies were excluded in favor of `main_9`.

The [paper coverage table](../provenance/PAPER_COVERAGE.csv), [numbering map](../provenance/NUMBERING_AND_REFERENCE_MAP.csv) and [manuscript correspondence](manuscript_mapping.md) identify related table/appendix roles. The publication release does not claim end-to-end historical figure rerunning.
