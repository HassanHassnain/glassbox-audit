# Result Matrix

| Block | Status | Main readout | Claim impact |
|---|---|---|---|
| Main Qwen2.5-1.5B audit | completed | layer 27; mean delta `-0.136`; SAE delta `-0.036` | residual claim supported; SAE superiority failed |
| SAE stability grid | completed | 6/6 fixed cells; 0 SAE-vs-baseline passes | negative SAE result strengthened |
| OR-Bench transfer | completed | SAE toxic refusal-rate delta `-0.12`; mean `-0.68` | transfer present, SAE not mean-superior |
| Qwen2.5-3B replication | partial | late layer 35, specificity failed | family replication only |
| Component/path | completed | residual strong; attention/MLP small or non-specific | no circuit claim |
| Clean-room rerun | completed | fixed-tolerance reproduction | release claim unchanged |
