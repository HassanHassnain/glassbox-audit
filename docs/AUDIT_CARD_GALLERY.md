# Audit Card Gallery

The C2 gallery renders B1 certification cards for the main audit and an intentionally failing
replication case.

Build it with:

```bash
python scripts/build_audit_card_gallery.py
```

Generated artifacts:

- `results/certification/qwen-real-audit.card.json`;
- `docs/audit-cards/qwen-real-audit.card.html`;
- `results/certification/qwen-real-audit.card.pdf`;
- `results/certification/qwen2.5-3b-replication.card.json`;
- `docs/audit-cards/qwen2.5-3b-replication.card.html`;
- `results/certification/qwen2.5-3b-replication.card.pdf`;
- `results/certification/gallery.json`.

The 3B card is expected to fail because the specificity rung does not pass.
