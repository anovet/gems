# Keep model predictions on known-fault pixels

Known faults are excluded from scoring in the Initial Prize Round, so it is tempting to zero out (or force to 1) our predictions on those pixels. We leave them exactly as the model predicts. One submission is scored in both rounds, and the Final Prize Round label set, which experts build after reviewing all submissions, almost certainly includes known faults. Zeroing them can at best help Phase 1, and only if the scorer counts them as false positives, which their exclusion suggests it doesn't. It would clearly hurt Phase 2.

Forcing known-fault pixels to 1 for Phase 2 is a reasonable later refinement. It is deferred, not rejected.
