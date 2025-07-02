## `fail_submission.py`

### Submission Summary

This version performs pruning but does not propagate effects or revisit related arcs:

```python
def gac_enforce(csp, queue=None):
    for var in csp.variables:
        for val in csp.get_domain(var)[:]:
            for c in csp.get_constraints_with(var):
                if not c.has_support(var, val):
                    csp.prune_value(var, val)
    return True, []
```
## Issues Identified

- There is no queue or arc processing logic.
- No handling of downstream effects when pruning changes domains.
- No check for domain wipeout (i.e., empty domains).

### Expected AI Response

- Identify that there is no propagation mechanism.
- Suggest using a queue to track and revisit arcs when pruning affects constraints.
- Point out missing early return on empty domains.

---
