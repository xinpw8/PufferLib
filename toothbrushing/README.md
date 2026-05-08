# Toothbrushing Tracker

Log of kid's toothbrushing sessions. Birthdate: **2020-01-01**.

## Log file

`log.csv` — one row per brushing.

## Columns

| column      | format                              | example          |
|-------------|-------------------------------------|------------------|
| date        | `YYYY-MM-DD`                        | `2026-05-08`     |
| clock_time  | `HH:MM` (12-hour)                   | `7:45`           |
| am_pm       | `AM` or `PM`                        | `PM`             |
| duration    | minutes:seconds or descriptive      | `2:30`           |
| rating      | 1-10 (10 = best cooperation)        | `7`              |
| age         | `<years>y<months>m` from 2020-01-01 | `6y4m`           |
| notes       | free text (no commas, or quote)     | `wiggly tooth`   |

## How to log a new entry

The user submits a prompt with the brushing info (date, time, duration, rating, notes).
Claude appends a row to `log.csv`, computing `age` from the date relative to 2020-01-01:
- years = floor of full years elapsed
- months = remaining whole months

If the date is omitted, use today. If notes contain a comma, wrap the field in double quotes.

## Quick stats

To eyeball recent entries: `tail log.csv`
To count by rating: `cut -d, -f5 log.csv | sort | uniq -c`
Average duration last 7 entries: `tail -7 log.csv | awk -F, '{print $4}'`
