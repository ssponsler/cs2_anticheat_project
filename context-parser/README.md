# Context Parser Pipelines

This folder contains standalone Go commands for extracting ML-ready events from CS2 demos.

## Pre-kill context-window dataset pipeline

Use `prekill_cw_pipeline.go` to ingest all `.dem` files under a root folder, run `prekill_cw.go` for each demo, and merge the outputs into one labeled dataset.

### Label conventions

Labels are inferred from directory names in the demo path:

- `cheater`, `cheaters`, `hacker`, `hackers`, `rage`, `spinbot` -> `label=1`, `label_name=cheater`
- `no-cheater`, `no_cheater`, `no-cheaters`, `no_cheaters`, `legit`, `clean`, `non-cheater`, `non_cheater` -> `label=0`, `label_name=no_cheater`
- otherwise -> `label=-1`, `label_name=unknown`

Recommended demo layout under `test/cs-demos`:

- `test/cs-demos/cheater/*.dem`
- `test/cs-demos/no_cheater/*.dem`

### Run

From repository root:

```bash
go run ./context-parser/prekill_cw_pipeline.go \
  -demos-root ./test/cs-demos \
  -include-seq=true \
  -jsonl-out ./context-parser/out/prekill_cw_dataset.jsonl \
  -csv-out ./context-parser/out/prekill_cw_dataset.csv
```

flags:

- `-include-seq=true` to include per-tick sequence channels in JSONL output
- `-include-unknown=true` to include demos that do not match label tokens
- `-continue-on-fail=true` to skip failed demos and continue
- `-keep-interim=true` to preserve per-demo temporary outputs
- `-verbose=true` for per-demo progress logs

### Output schema additions

Each merged row includes the original `prekill_cw` fields in addition to:

- `demo_path`
- `demo_name`
- `label`
- `label_name`

When `-include-seq=true` is enabled, JSONL rows also include per-tick channels:

- `seq_ang_offset_deg`
- `seq_spotted_flag`
- `seq_killer_speed`
- `seq_killer_scoped_flag`
- `seq_victim_audible_any_flag`

## Build fixed-length input tensors

Use `build_input_tensors.go` to convert the merged JSONL into fixed-length sequence tensors (`X_seq`) and global side features (`X_global`).

From repository root:

```bash
go run ./context-parser/build_input_tensors.go \
  -in-jsonl ./context-parser/out/prekill_cw_dataset.jsonl \
  -out-jsonl ./context-parser/out/player_kill_window_tensor_v1.jsonl \
  -seq-len 192
```

Notes:

- The command expects sequence channels in input rows (generate with `-include-seq=true`).
- By default, unknown labels (`label=-1`) are skipped; use `-keep-unknown=true` to retain them.
- If scraper-joined fields are present, the builder encodes:
  - `map` -> `map_id` (categorical ID)
  - `weapon` -> `weapon_id` (categorical ID)
  - `avg_rank_raw` -> `avg_rank_numeric` + `avg_rank_available`
- NumPy export is enabled by default (`-export-npy=true`) for direct Python training.
- Use `-out-npy-dir` to control where `.npy` files are written.
- Convenience NPZ bundle is enabled by default (`-export-npz=true`).
- Use `-out-npz` to control the `.npz` file path.

### NumPy outputs

The tensor builder writes these files for direct loading via `numpy.load(...)`:

- `x_seq.npy` shape `[N, T, F]` (float32)
- `x_global.npy` shape `[N, G]` (float32)
- `x_seq_mask.npy` shape `[N, T]` (uint8)
- `y.npy` shape `[N]` (int64)
- `player_kill_window_tensor_v1.npz` convenience archive containing `x_seq.npy`, `x_global.npy`, `x_seq_mask.npy`, `y.npy`, and `metadata.json`
- `map_vocab.json` mapping from normalized map names to `map_id`
- `weapon_vocab.json` mapping from normalized weapon names to `weapon_id`
- `sample_ids.json` sample identifiers aligned to array row order
- `metadata.json` shapes + feature names
