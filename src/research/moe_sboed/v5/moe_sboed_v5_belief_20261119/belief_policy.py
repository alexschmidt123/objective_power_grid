"""Fixed forward-only V5 actor. Belief snapshot refresh exists only in offline training code."""
from history_policy import V5Actor as HistoryActor
class V5Actor(HistoryActor):
    pass
