# Statistics QA — PASS

Independent synthetic fixtures; PCG64(818181), separate from production streams. No real outcomes inspected for QA choices.

- separable: PASS
- reverse: PASS
- all_tied: PASS
- threshold_tie: PASS
- AURC original-index ties: PASS
- weighted AURC versus explicit expansion: PASS
- weighted ROC versus explicit expansion: 74 draws: PASS
- class counts and total counts: PASS
- paired source-index expansion: PASS

AUROC half-credit ties; OOD-positive full tied threshold at first TPR >= .95; no interpolation. AURC original-index tie order, weighted expansion, pairing and class counts verified. Actual values: audit/STATISTICS_QA.json. Statistics definitions unchanged.
