# Measured Slerp substitution closes this composer fixture

Replacing only the native Slerp callback with exact results measured from the original compiled REK/Unity functions removes every supported-field difference across the unchanged 640-row fixture.

| Quantity | Native baseline | Measured-callback replay |
|---|---:|---:|
| Differing root-quaternion components / 12,800 | 6,150 | 0 |
| Maximum root-quaternion component error | 2.7939677 × 10⁻⁶ | 0 |
| Differing joint references / 92,800 | 0 | 0 |
| Differing compared float values / 129,280 | 6,150 | 0 |
| Differing compared integer values / 38,400 | 0 | 0 |

All 32,800 callback invocations used an exact nine-word input match and a measured original output. There were zero missing tuples and zero fallbacks in the final replay. The final oracle measured 513 inputs twice; both repetitions and both original wrappers agreed bit for bit. Actual replay used 308 distinct inputs from that table.

The native baseline capture is byte-identical to the previous native trace. This establishes the Slerp callback boundary as the cause of the measured quaternion residual in this explicit fixture. The lookup table is a diagnostic for covered inputs, not a general Slerp implementation or a production correction.

The first private oracle run produced zero math rows and rejected an ambiguous reflection overload. The corrected plugin selects and records exact parameter types. The initial measured-table replay then stopped at call 3,002 when original interpolation changed a later blend input. One bounded follow-up queried 199 additional candidate inputs. All failures and partial evidence remain preserved. A zero process exit alone did not imply oracle success.

All three task containers exited. They had fresh profiles, no network, no GPU access and no host display. The installed game, viewer and native clone were not modified by this experiment. The parent task owns the separate before/after viewer and worker identity check.

The comparison excludes unavailable original `has_clip`, unimplemented native reference root positions and the native velocity estimate as an API-parity claim. It uses the staged original-semantics reset extension with both explicit reset orders. It does not establish contact physics, server behavior, sparring strength or general environment parity. The comparator's 4.21 × 10⁻⁸ rad orientation estimate on identical quaternion words is dot/norm/acos roundoff, not a remaining quaternion discrepancy.

`RESULTS.json`, `COMPARISON.json`, exact source/asset manifests, closed original traces and `review/CLOSED-RESULT-REVIEW.json` provide the reproducible evidence.
