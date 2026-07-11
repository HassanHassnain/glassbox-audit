# Paper hardening figure contract

## Feature-budget frontier

- Question: How many train-ranked SAE features are needed to match the mean-difference ablation, and do they beat same-dictionary random latent sets?
- Expected takeaway: The selected SAE frontier is non-monotonic and remains below the mean-direction reference; uncertainty/control bands prevent over-reading individual budgets.
- Family/variant: ordered-axis line with point markers, matched-random min–max band, and a horizontal benchmark.
- Data: seven predeclared budgets (1, 2, 5, 10, 20, 50, 100), one selected set and eight matched random sets per budget, all on 250 harmful records from the frozen test split.
- Renderer/surface: reproducible Matplotlib static PNG and PDF for the paper.
- Palette: single blue root for selected SAE, neutral grey control band, gold benchmark; markers, band fill, and dashed reference remain distinct without color.

## Intervention comparability

- Question: How much refusal suppression does each intervention buy for its hidden-state perturbation and capability cost?
- Expected takeaway: Full SAE substitution is dominated by reconstruction distortion; residual-preserving feature subtraction is much less destructive but also much weaker than the mean direction.
- Family/variant: directly labeled scatter of harmful-score suppression versus mean hidden-state L2, with marker area representing positive NLL cost.
- Data: seven intentionally selected method-level observations on the same 500-record frozen test split. Although fewer than a generic exploratory scatter, every point is a named experimental arm and is directly labeled.
- Renderer/surface: reproducible Matplotlib static PNG and PDF for the paper.
- Palette: blue for residual-preserving/linear methods, gold for full SAE substitution/reconstruction, neutral reference; marker shape plus labels provide non-color distinction.
