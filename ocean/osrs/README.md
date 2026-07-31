# OSRS encounters

## Build

```bash
./build.sh osrs_colosseum
./build.sh osrs_inferno
```

## Train and eval

```bash
./puffer train osrs_colosseum
./puffer eval osrs_colosseum
./puffer eval osrs_colosseum --load-model-path=checkpoints/osrs_colosseum/model.bin
./puffer sweep osrs_inferno
```

## Tests

```bash
cc -std=c11 -O2 -I. -o /tmp/t ocean/osrs/tests/test_colosseum_golden.c -lm && /tmp/t
```
