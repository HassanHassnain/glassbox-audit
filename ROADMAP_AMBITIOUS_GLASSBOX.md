# Glassbox v2 — the ambitious roadmap

Spec-grade briefs for an LLM implementer. Each item: what to build, why reviewers/companies care, how to build it, and the artifact that proves it. The north star stays fixed: **Glassbox is the harness that decides whether interpretability claims survive preregistered, held-out, baseline-anchored causal tests.** Every extension below compounds that identity.

Ordering inside each tier ≈ impact per unit of risk.

---

## Tier A — Science that makes reviewers sit up

### A1. Audit the public SAE zoo (Gemma Scope, Llama Scope) — *the single highest-value extension*

**What.** Re-run the identical audit, but instead of only home-trained SAEs, plug in the strongest public dictionaries: DeepMind's **Gemma Scope** (JumpReLU SAEs for Gemma-2-2B/9B, all layers, widths up to ~131k) and **Llama Scope** (Llama-3.1-8B). Same walls, same baselines, same controls.

**Why it's "insane".** Right now a reviewer can say "your SAE was small and home-trained." If *Gemma Scope's* 131k-width dictionaries also fail to beat a mean direction on held-out causal tests, the negative result stops being a case study and becomes a field-level finding. If they *win*, the paper becomes "here is the scale/architecture where SAE features start earning their keep" — a positive scaling result. **Either outcome upgrades the paper.** This also converts H5 (Gemma recipe) from "unrun" to "run with the best available dictionaries."

**How.** Add an `sae.source: {trained | saelens}` config path; load checkpoints via SAELens; map each public SAE to its layer; keep feature selection train-only (rank public latents by train effect size — no cherry-picking from feature dashboards). Matched random-SAE controls draw random latent sets from the same public dictionary. New evidence file: `results/sae-zoo/<model>_<sae_id>.json`.

**Artifact.** A "dictionary league table": rows = SAE checkpoints (width, architecture, layer), columns = held-out Δ, control range, beats-mean?, beats-probe?. One figure, one table, one sentence in the abstract.

### A2. The negative result vs scale: a 10-model sweep

**What.** Run the full audit on ~10 open instruct models spanning 0.5B→14B (Qwen2.5 0.5/1.5/3/7/14B, Llama-3.2-1B/3B, Llama-3.1-8B, Gemma-2-2B/9B, Mistral-7B). Plot (a) refusal-direction effect size vs scale, (b) the SAE-minus-mean gap vs scale, (c) target-layer depth fraction vs scale.

**Why.** Reviewers forgive a narrow behavior; they don't forgive a single model. A monotone (or non-monotone!) trend line turns anecdote into law: *"the refusal direction is a universal late-depth structure; the SAE gap does/doesn't close with scale."* Also fixes the current 3B embarrassment (n=60, half-dead SAE) by making cross-model the core design rather than an afterthought.

**How.** The harness already takes model configs; the work is (1) per-model corpus re-scoring thresholds on validation, (2) SAE budget scaled by hidden size (fixed samples-per-feature, so cells are comparable — preregister this rule), (3) a sweep driver `scripts/sweep_models.sh` writing one evidence JSON per model, (4) n ≥ 250 held-out records per model (fix the 3B n=60 problem). Preregister the whole grid *before* running any cell; publish every cell.

### A3. When does the refusal direction form? (developmental audit on Pythia/OLMo checkpoints)

**What.** Run the audit across public *training checkpoints* (Pythia suite and OLMo release intermediate checkpoints, plus their instruct-tuned endpoints where available): at which point in training does a causally effective refusal/compliance direction emerge, and does it move layers?

**Why.** Nobody has a preregistered causal account of *when* safety-relevant structure forms during training. This is the "how did he do that" experiment — it looks expensive but is embarrassingly parallel and uses only public checkpoints. Even 6 checkpoints × 2 models is a novel figure: effect size vs training tokens.

**How.** Base (non-instruct) checkpoints won't refuse; audit the *harmful/benign representation separation* (H1's separation rung + probe/mean direction transfer INTO the final instruct model — "was the direction already there before RLHF?"). Concretely: compute mean-diff direction at checkpoint t, freeze it, causally test it on the *final* instruct model. If early-pretraining directions already steer the final model's refusal, that is a headline finding about where safety behavior lives.

### A4. The price of sparsity: feature-budget frontier

**What.** Sweep the SAE intervention budget: top-k latents for k ∈ {1, 2, 5, 10, 20, 50, 100, 500, 1000}, plus an ℓ1-regressed *weighted* latent combination (best-case SAE handle at each budget). Plot held-out Δ vs budget against the flat 1-dim mean-direction line.

**Why.** Converts the binary verdict into an exchange rate: *"the dictionary needs ~N latents to buy what one mean direction buys."* Reviewers love a frontier; it also preempts "you only used 5 features." If the curve crosses the mean line at k=40, that's an interesting positive fact about distributed encoding; if it plateaus below, the negative result becomes quantitative.

**How.** Feature ranking and the ℓ1 regression are train-only; budgets preregistered; one new stage in `pipeline.py` + one figure. Cheap (reuses cached activations); very high information density.

### A5. Settle H4 with head-level path patching

**What.** Upgrade component analysis to head granularity: attribution patching (gradient-approximated) to shortlist heads/paths writing along the refusal direction, then exact activation patching on the shortlist, with the same held-out discipline. Either find the writing circuit or upgrade "no circuit found" to "no circuit exists at head granularity under these tests."

**Why.** H4 currently *fails for lack of evidence* — the weakest rung methodologically. Reviewers will ask. Answering with a two-stage attribution→verification pipeline (the field-standard trick for tractability) shows technical depth, and "which heads write the refusal direction" is independently interesting.

**How.** Implement attribution patching over head outputs projected on $v_{mean}$; take top-20 candidates; run true patching on held-out records with BH correction; preregister the necessity/sufficiency thresholds (reuse H4's existing bundle). New evidence file + one circuit-diagram figure (or an honest "no head survives" table).

### A6. Equivalence testing + power analysis (the statistical upgrade reviewers rarely see)

**What.** (1) Preregistered **TOST equivalence tests**: "SAE is not better" is currently absence-of-superiority; add explicit equivalence/inferiority bounds (e.g., SAE effect within ±δ of probe; inferior to mean by margin ≥ δ*) so the negative claim is affirmatively powered. (2) Power analysis justifying n=500 for the preregistered effect sizes. (3) Hierarchical bootstrap over *pairs* (not records) to respect the paired design.

**Why.** This is the difference between "we didn't see it" and "we can rule it out at margin δ with power 1−β." Almost nobody in interp does this; statisticians on the review panel will notice. Small implementation, outsized credibility.

### A7. Second behavior: sycophancy (prove the harness is behavior-agnostic)

**What.** Clone the whole audit for **sycophancy** (agreement flips under user pressure): paired prompts (same question ± user's stated opinion), contrastive scorer on agree/disagree prefixes, same discovery/selection/judgement walls, same baselines.

**Why.** One behavior = a refusal paper. Two behaviors with the same protocol = a *methodology* paper (the stronger NeurIPS frame: "a preregistered causal audit protocol, instantiated on two safety-relevant behaviors"). Sycophancy is topical, benign to publish, and has public eval sets to adapt.

**How.** The corpus builder already supports pairing; the scorer is a prefix swap; the only new work is the dataset (LLM-generate + human-spot-check 1,000 pairs, leakage-audit them with the existing tool).

### A8. Does the direction survive adversarial pressure? (robustness of the localization claim)

**What.** Test whether the layer-27 direction remains the mediator under distribution shift designed to stress it: paraphrase attacks, persona framings, multilingual prompts, and standard jailbreak formats — measuring *aggregate* refusal-score mediation (does ablation still move the score the same way?), not publishing any bypass content.

**Why.** "Is it one direction or many?" under adversarial shift is exactly what a safety team needs to know before trusting direction-based mitigations. Frame strictly as robustness-of-mediation; report aggregate deltas only. (Keep this one carefully scoped — evaluation of a published mechanism, no attack tooling in the repo.)

---

## Tier B — The industry story: Glassbox as infrastructure companies use

### B1. `glassbox certify` — audit cards for interventions (the flagship)

**What.** A CLI that takes: any HF model + a behavior pair-file + a *candidate intervention* (a direction tensor, a latent set + SAE checkpoint, or a steering vector) and runs the frozen protocol end-to-end, emitting a signed **Intervention Audit Card**: JSON + rendered HTML/PDF stating exactly which rungs passed (separation → held-out causal → specificity → capability cost → transfer), with CIs, controls, and hashes.

**Why companies care.** Labs and enterprises are starting to ship steering-based mitigations and buy interpretability tooling. Today there is no standard way to *accept-test* a claimed intervention. Model cards standardized model reporting; audit cards do it for interventions. This is the artifact a safety team attaches to a deployment review. For the paper: one paragraph + an appendix card = "this is very helpful" reaction.

**How.** 90% exists (pipeline + report + release_hardening). Work: a clean `certify` subcommand, an intervention-loading interface (`.pt` direction / SAELens latents / callable), a card schema (`schema_version`, model hash, corpus hash, prereg hash, per-rung verdicts), an HTML template in the house style, and `--sign` (hash-chain the card against the git tag).

### B2. Mechanism-drift monitoring across model versions

**What.** `glassbox drift`: re-evaluate *frozen* artifacts (directions, latents, thresholds) against a new checkpoint of the "same" model (v2.5 → v3, base → RLHF-updated, pre/post fine-tune) and report which rungs still pass — i.e., **does the safety-relevant mechanism survive the update?**

**Why.** Every deployed steering/probing mitigation rots silently when the base model updates. Nobody ships tooling for this. It's cheap (frozen artifacts are already the repo's design principle — the OR-Bench transfer *is* a drift test across data; this is drift across weights). Enterprise pitch: "add to CI; know the day your refusal direction stops meaning refusal."

**How.** New subcommand reusing external-causal machinery with a model swap instead of a data swap; a drift report comparing rung verdicts old-vs-new; demo it on Qwen2.5-1.5B vs Qwen2.5-3B (already have the data) and one LoRA fine-tune you produce yourself.

### B3. A public causal-handle leaderboard (community magnet)

**What.** A static GitHub Pages leaderboard: rows = (model, behavior, method/SAE checkpoint), columns = held-out Δ, specificity, capability cost, verdicts — populated by CI from `results/`, open to PR submissions that include the evidence JSON + artifact hashes (the release-check script becomes the submission validator).

**Why.** Leaderboards are how a harness becomes a standard (and how a solo repo accumulates citations and contributors). SAEBench measures SAE *quality*; this ranks *causal utility per behavior* — complementary, not competing.

**How.** `scripts/build_leaderboard.py` → JSON → single static HTML (house style, no backend); CONTRIBUTING.md defines the submission bar (preregistered config, all cells published, hashes pinned).

### B4. Backend + ecosystem integrations

**What.** (1) `nnsight`/`TransformerLens` backends beside the raw HF hooks, (2) SAELens loader (needed for A1 anyway), (3) `pip install glassbox-audit` on PyPI with the toy audit as the smoke test, (4) a Colab/Modal "reproduce the headline in ~40 min for <$10" notebook.

**Why.** Adoption is a review criterion in practice: tools reviewers can run become tools reviewers cite. The Colab is disproportionately powerful — a reviewer who reproduces your headline number becomes your advocate.

### B5. Preregistration-as-code (tiny build, huge signal)

**What.** `glassbox prereg freeze`: hypotheses/thresholds/run-order in schema-validated YAML; freezing computes a hash, embeds it in every subsequent evidence file, and refuses to run cells whose criteria hash differs from the frozen one. Tamper-evident preregistration for interpretability experiments.

**Why.** Turns the repo's social promise ("we didn't retune") into a mechanical guarantee. No interp tooling does this. One-day build; reviewers and safety teams both love it; it generalizes beyond this project.

---

## Tier C — Force multipliers on presentation

- **C1. Interactive evidence explorer.** Static HTML (GitHub Pages, house style): click a ladder rung → every cell, CI, and control draw behind it. The claim ladder becomes navigable evidence.
- **C2. Audit-card gallery in the README.** Render B1 cards for the main audit and the 3B failure — showing the harness *failing loudly* is the credibility flex.
- **C3. A 90-second narrated demo video/GIF** of `glassbox certify` running end-to-end on a small model.
- **C4. Docs site** (mkdocs-material, dark theme matching the identity) with a "write your own behavior audit in 30 minutes" tutorial — the tutorial is what converts companies.

---

## Suggested execution order (dependencies in parentheses)

1. **A6** equivalence tests + **B5** prereg-as-code — small, harden the core first
2. **A4** feature-budget frontier — cheap, reuses cached activations
3. **A1** SAE-zoo via SAELens (needs B4-partial) — the headline upgrade
4. **A2** 10-model sweep (subsumes the old "run Gemma/Llama recipes" advice)
5. **A5** head-level path patching — settle H4
6. **A7** sycophancy behavior — reframe as protocol paper
7. **B1** certify + audit cards → **B2** drift → **B3** leaderboard → **C1–C4**
8. **A3** developmental checkpoints — the moonshot; start it in parallel, it's pure compute
9. **A8** adversarial mediation robustness — last, with careful scoping

**Definition of "insane" (acceptance bar).** The paper can claim: *a preregistered causal audit protocol; instantiated on two behaviors; across ten models and the strongest public SAEs; with equivalence-powered negative results, a settled circuit question, and a certification tool that any lab can point at its own intervention tomorrow.* Every cell published, every number traceable, every failure kept.

**Standing rules for the implementer.** Preregister before running; publish every predeclared cell; every new evidence file gets a stable name, a schema version, and a figure-footer citation; no experiment ships without its matched control; aggregate metrics only for anything adversarial; the README claim ladder is updated the day a rung changes.
