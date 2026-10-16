# C3 Certify Demo

The C3 demo is a 90-second captioned MP4 of `glassbox certify` rendering a signed Intervention Audit
Card from the committed toy audit artifacts.

Build it with:

```bash
python scripts/build_certify_demo.py
```

or:

```bash
make certify-demo
```

Generated assets:

- `docs/demo/certify-demo.mp4`;
- `docs/demo/certify-demo.vtt`;
- `docs/demo/certify-demo-transcript.md`;
- `docs/demo/certify-demo.storyboard.json`;
- `docs/demo/certify-demo.stdout.txt`;
- `docs/demo/certify-demo.card.html`;
- `results/certification/toy-certify-demo.card.json`;
- `results/certification/toy-certify-demo.card.pdf`.

The demo uses `artifacts/demo` so it stays fast and works without a GPU. It intentionally shows a
failing SAE feature-ablation card: the certifier accepts the frozen evidence, scores each required
rung, and rejects the intervention because held-out causal effect and matched controls fail.
