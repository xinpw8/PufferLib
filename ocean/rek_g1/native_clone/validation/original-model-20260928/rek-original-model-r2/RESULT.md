# Original model inventory revision 2

The one authorized r2 passive run completed successfully: container exit 0, no OOM, and inventory_end.success=true. All 773 JSON records are complete. Closed source/readback verification covers 23 files and 687,997 bytes.

The exact original catalog was loaded and G1 selected by unique observed id through direct catalog enumeration. The unsafe generated TryGetById invocation is absent; its original semantics were not validated. The immutable first failed run and its independent ABI review are linked in RESULT.json and preserved in the source package.

The prefab inventory contains 226 transforms and 498 components, including all 30 MuJoCo bodies, 30 inertials, 29 hinge joints, one free joint, 37 geoms and 29 actuators identified by the independent static asset audit. It includes original Transform TRS, masses/inertias, hinge limits/armature/damping/friction, geom shape/contact fields, actuator fields, and serialized controller fields. It also captured two RobotConfig assets and 36 MocapClipConfig assets. The 201 JsonUtility records total 141,465 serialized bytes and agree with the footer.

One loaded MJSettings component belongs to the Arena scene. Its serialized options include integrator 3, cone 1, Jacobian 2, solver 2, iterations 100 and tolerance approximately 1e-8. These are actual serialized observations, not effective compiled mjModel options. The application loaded that scene during its normal isolated startup; the harness made no scene-load call.

Before and after inspection, no Robot, SonicPolicyRunner or SonicMotionComposer scene instance was found. Their asset counts increased as expected after Resources.Load. One active MjScene was already present at the first count. Whole-process physics activity was not instrumented.

This completes the serialized dependency inventory, not initialization/model parity. ONNX/config fields on the uninitialized runner prefab are zero; that does not imply missing shipped weights. References serialized as 64-bit identifiers are not silently equated with 32-bit GetInstanceID outputs. The next model export must establish original runtime wiring, generated model, effective options and initial mjData/controller history. No model creation, robot spawn, dynamics or further runtime was performed in this pass.

