# Toothbrushing Tracker

Log of kid's toothbrushing sessions. Birthdate: **2020-01-01**.

## Routine (lap-timed)

1. Brush teeth   → *lap*
2. Brush tongue  → *lap*
3. Floss

Each segment is captured as its own duration. `total` is the sum.

## Log file

`log.csv` — one row per brushing.

## Columns

| column      | format                              | example          |
|-------------|-------------------------------------|------------------|
| date        | `YYYY-MM-DD`                        | `2026-05-08`     |
| clock_time  | `HH:MM` (12-hour)                   | `7:45`           |
| am_pm       | `AM` or `PM`                        | `PM`             |
| teeth       | `M:SS` brushing teeth               | `1:30`           |
| tongue      | `M:SS` brushing tongue              | `0:20`           |
| floss       | `M:SS` flossing                     | `0:40`           |
| total       | `M:SS` sum of segments              | `2:30`           |
| rating      | 1-10 (10 = best cooperation)        | `7`              |
| age         | `<years>y<months>m` from 2020-01-01 | `6y4m`           |
| notes       | free text (no commas, or quote)     | `wiggly tooth`   |

## How to log a new entry

User submits a prompt with the lap times, clock time, rating, notes.
Claude appends a row to `log.csv`, computing:
- `total` = teeth + tongue + floss
- `age` from the date relative to 2020-01-01 (full years + remaining whole months)

If a segment was skipped, leave the field empty (e.g. `,,` for no flossing).
If notes contain a comma, wrap the field in double quotes.

## Quick stats

Recent entries: `tail log.csv`
Avg total over last 7: `tail -7 log.csv | awk -F, '{print $7}'`
Rating histogram: `cut -d, -f8 log.csv | tail -n +2 | sort | uniq -c`
