# Original REK model dependency inventory

This package preserves the failed first passive inventory and successful reviewed revision. No model export, robot instantiation, reset, dynamics experiment or authoritative-server validation occurred in either harness.

- [First run](rek-original-model-r1/RESULT.md): six valid records, then an access violation through the generated value-type out wrapper. The original failed trace and receipts remain unchanged.
- [Second run](rek-original-model-r2/RESULT.md): unique G1 selection from the original catalog, 226 transforms, 498 components, 201 serialized records, and an explicit successful footer. Selection does not validate original TryGetById semantics.
- The independent ABI diagnosis is retained in the second source package at `rek-original-model-r2/package/review/closed-inventory-independent-review-r1.json`. The final independent closed-result review is in `review-supplement/`.

Both nested PUBLICATION.json receipts retain their original NAS manifests. Their historical sibling names resolve in this directory. The top-level MANIFEST.json pins every file in this combined package, including the separate review supplement. Source asset hashes, original library hashes and reproduction tooling are included; original game assets and profiles are not copied into this package.

The serialized prefab and loaded Arena settings are established. Effective compiled mjModel options, initialized mjData, controller history, lifecycle overrides and server selection remain unverified. JSON instance references and GetInstanceID values remain separate observations.
