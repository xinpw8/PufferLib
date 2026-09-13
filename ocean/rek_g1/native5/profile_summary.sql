-- Run with native sqlite3 against a private Nsight SQLite export.
-- Export aggregate kernel/API measurements only; never publish raw traces.
.headers on
.mode csv
SELECT s.value AS kernel, COUNT(*) AS calls,
       ROUND(SUM(k.end-k.start)/1e6,6) AS total_ms,
       ROUND(100.0*SUM(k.end-k.start)/(SELECT SUM(end-start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL),6) AS gpu_kernel_pct,
       MIN(k.gridX) AS min_grid_x, MAX(k.gridX) AS max_grid_x,
       MIN(k.blockX) AS min_block_x, MAX(k.blockX) AS max_block_x,
       MAX(k.registersPerThread) AS registers_per_thread
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC;

SELECT s.value AS api, COUNT(*) AS calls,
       ROUND(SUM(k.end-k.start)/1e6,6) AS total_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s ON s.id=k.nameId
GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC;

SELECT COUNT(*) AS kernel_calls,
       ROUND(SUM(end-start)/1e9,6) AS kernel_seconds,
       ROUND((MAX(end)-MIN(start))/1e9,6) AS device_span_seconds
FROM CUPTI_ACTIVITY_KIND_KERNEL;
