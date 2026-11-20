# Research focus restored

V4 and previous history-conditioned V5 are the active architecture references. V4 source:src/research/moe_sboed_v4_20261116/. V5 source:src/research/moe_sboed_v5_history_20261119/. Previous V5 canonical seed101 learned/uniform:12.89320/12.86427;V4 same-seed controls:12.85611/12.80587. V4 five-seed means remain separate.

Native critic and posterior-training machinery stay unchanged. Future changes are confined to the four-expert MoE policy and its existing history/routing/fusion components. No additional external networks. Belief-refresh and critic variants are excluded from proposed-method development;all artifacts are preserved. The currently running critic-uniform job is not interrupted and no new training is launched.
