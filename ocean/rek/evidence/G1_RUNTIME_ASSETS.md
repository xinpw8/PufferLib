# G1 runtime motion assets

`g1_asset_extract.py` copies the original G1 motion TextAsset payloads from the
installed REK build into a new local directory. It does not modify the game or
write extracted payloads into this Git work tree.

The allowlist is build-bound by all of the following:

- REK build fingerprint
  `f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659`
- `sharedassets0.assets` byte count and SHA-256
- Unity object path ID, exact TextAsset name, byte count, and SHA-256
- exact NPZ member names, byte counts, and SHA-256 values
- UnityPy version `1.25.2`

The 21 payloads are all 17 entries in the pinned G1 `RobotConfig.moves` array
plus the four unique processed locomotion references: walk, idle, left strafe,
and right turn. Each move row records its exact array index and referenced
`MocapClipConfig` path ID. The pinned build does not provide separately named
processed right-strafe or left-turn TextAssets. Its serialized configuration
references the left-strafe and right-turn payloads with negative playback speed
for the opposite directions. The extractor preserves only the original payload
bytes and does not synthesize a mirrored trajectory.

Move-array membership is static asset evidence. It does not establish input
reachability, active server behavior, hit semantics, or trajectory parity.
Input bindings are outside this asset manifest. The pinned static keyboard
scheme does not bind move indices 12 or 13; reachability through any other input
surface remains unasserted.

Install the pinned parser, then select a new output directory outside every Git
work tree:

```powershell
python -m pip install UnityPy==1.25.2
python .\g1_asset_extract.py `
  --assets "C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK_Data\sharedassets0.assets" `
  --out-dir "D:\REK-local-assets\g1-f84f1874"
```

The command validates every payload in memory before publishing anything. The
output directory must not already exist. On success it contains 21 byte-exact
`.npz` files and `g1_runtime_assets.inventory.json`. The inventory contains no
timestamp or host-specific input path, so an extraction of the same pinned build
is deterministic.

The files contain reference trajectories. Their presence does not establish how
REK blends, mirrors, phase-aligns, accepts, or executes them. Those transition
semantics still require runtime evidence.

## Plant boundary

The matching recovered plant is `evidence_out/g1_29dof.recovered.xml`, SHA-256
`811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835`.
Its report records 30 bodies, 29 hinge joints, 29 actuators, `nq=36`, `nv=35`,
and `nu=29`, with no unresolved actuators. The same report explicitly marks the
plant `control_equivalent: false`. Supplying these motion payloads does not
change that evidence boundary or establish trajectory parity.

Run the fixture and manifest tests without an installed game or UnityPy:

```powershell
python -m unittest -v test_g1_asset_extract.py
```
