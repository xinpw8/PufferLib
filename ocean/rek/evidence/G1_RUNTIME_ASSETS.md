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

The eight payloads are the four G1 kicks plus the processed walk, idle, left
strafe, and right turn references. The pinned build does not provide separately
named processed right-strafe or left-turn TextAssets. This extractor does not
mirror a trajectory or infer that one reference implements the opposite
direction.

Install the pinned parser, then select a new output directory outside every Git
work tree:

```powershell
python -m pip install UnityPy==1.25.2
python .\g1_asset_extract.py `
  --assets "C:\Program Files (x86)\Steam\steamapps\common\REK Alpha Test\REK_Data\sharedassets0.assets" `
  --out-dir "D:\REK-local-assets\g1-f84f1874"
```

The command validates every payload in memory before publishing anything. The
output directory must not already exist. On success it contains eight byte-exact
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
