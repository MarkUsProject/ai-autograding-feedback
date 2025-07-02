## `partial_correct_submission.py`

### Submission Summary

Includes a queue-based structure but does not re-add arcs after pruning:

```python
def gac_enforce(csp, queue=None):
    if queue is None:
        queue = [(c, var) for c in csp.constraints for var in c.scope]

    pruned = []
    while queue:
        c, var = queue.pop(0)
        for val in csp.get_domain(var)[:]:
            if not c.has_support(var, val):
                csp.prune_value(var, val)
                pruned.append((var, val))
    return True, pruned
```
## Issues Identified

- Performs pruning, but pruned values don’t trigger updates to related arcs.
- No failure handling for empty domains.

### Expected AI Response

- Acknowledge use of queue but highlight lack of propagation.
- Recommend re-adding related arcs when pruning happens.
- Mention that empty domains should return failure.

---
