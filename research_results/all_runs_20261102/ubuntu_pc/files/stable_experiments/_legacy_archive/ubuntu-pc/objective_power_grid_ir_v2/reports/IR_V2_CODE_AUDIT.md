# Code audit

Published IR already contains the necessary history-gradient correction; it is preserved. The source report writer has a Path-plus-string error, so v2 uses a separate lifecycle and evaluator. Native GPU tests verify all components. Auxiliary-only expert updates are included in skip/Adam accounting, preventing valid expert gradients from being masked. Conditional information uses independent posterior banks, not hidden generating theta. Full remaining-horizon and exact gradient/resumption tests passed. Old worktrees are unchanged.
