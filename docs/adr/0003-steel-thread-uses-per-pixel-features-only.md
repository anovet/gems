---
status: accepted (V1 only; revisit in V2)
---

# Steel thread uses per-pixel feature bands only — a known, deliberate limitation

V1 trains a gradient-boosted tree classifier on the 19 raw feature bands of each pixel in isolation, with no neighbourhood context, so that the steel thread stays as simple as possible. **This is a known weakness, not a design conclusion.** A fault trace is a *linear* feature, and a single pixel cannot see a line, a lineament, or a scarp. Anyone proposing improvements should raise this as the first candidate.

## Planned direction

- **V2:** multi-scale neighbourhood features: local gradients, edge and lineament filters, and focal statistics over about 300 m, 1 km and 3 km.
- **V3:** a model that learns spatial context directly, such as a U-Net like the reference solution, compared against V2.
