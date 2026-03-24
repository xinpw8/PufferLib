"""
Live training dashboard server for pfr_native.
Serves heatmap.png and stats.json from /tmp/pfr_dashboard/
Auto-refreshes every 5 seconds.
"""
import http.server
import json
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 53580
DASH_DIR = "/tmp/pfr_dashboard"

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>pfr_native training</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #e0e0e0; font-family: 'Courier New', monospace; }
  .header { padding: 12px 20px; background: #16213e; border-bottom: 2px solid #0f3460; }
  .header h1 { font-size: 18px; color: #e94560; display: inline; }
  .header span { color: #888; font-size: 13px; margin-left: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 340px; gap: 0; height: calc(100vh - 48px); }
  .heatmap-panel { padding: 12px; overflow: auto; display: flex; align-items: center; justify-content: center; }
  .heatmap-panel img { max-width: 100%; max-height: calc(100vh - 72px); image-rendering: pixelated; border: 1px solid #333; }
  .zoom-panel { padding: 12px; overflow: auto; display: flex; flex-direction: column; align-items: center; border-left: 2px solid #0f3460; }
  .zoom-panel h3 { color: #e94560; font-size: 13px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
  .zoom-panel img { max-width: 100%; max-height: calc(100vh - 80px); image-rendering: pixelated; border: 1px solid #333; }
  .stats-panel { padding: 16px; background: #16213e; overflow-y: auto; border-left: 2px solid #0f3460; }
  .stat-group { margin-bottom: 16px; }
  .stat-group h3 { color: #e94560; font-size: 13px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
  .stat-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 14px; }
  .stat-row .label { color: #aaa; }
  .stat-row .value { color: #fff; font-weight: bold; }
  .stat-row .value.good { color: #4ade80; }
  .stat-row .value.warn { color: #fbbf24; }
  .stat-row .value.bad { color: #f87171; }
  .map-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; }
  .map-table th { text-align: left; color: #e94560; padding: 2px 4px; border-bottom: 1px solid #333; }
  .map-table td { padding: 2px 4px; border-bottom: 1px solid #222; }
  .no-data { color: #666; font-style: italic; text-align: center; padding: 40px; }
</style>
</head>
<body>
<div class="header">
  <h1>pfr_native</h1>
  <span id="update-time">waiting for data...</span>
</div>
<div class="grid">
  <div class="heatmap-panel" id="heatmap-panel">
    <div class="no-data">Waiting for first heatmap...</div>
  </div>
  <div class="zoom-panel" id="zoom-panel">
    <h3>Pallet → Pewter</h3>
    <div class="no-data">Waiting...</div>
  </div>
  <div class="stats-panel" id="stats-panel">
    <div class="no-data">Waiting for stats...</div>
  </div>
</div>
<script>
let lastStatsTime = 0;
function fmt(n) {
  if (n === null || n === undefined) return '-';
  if (typeof n === 'number') {
    if (Math.abs(n) >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return (n/1e3).toFixed(1) + 'K';
    if (Number.isInteger(n)) return n.toString();
    return n.toFixed(3);
  }
  return String(n);
}
function fmtTime(s) {
  if (!s) return '-';
  let m = Math.floor(s/60), sec = Math.floor(s%60);
  if (m >= 60) { let h = Math.floor(m/60); m = m%60; return h+'h '+m+'m'; }
  return m+'m '+sec+'s';
}
function classify(key, val) {
  if (key === 'entropy') return val > 1.5 ? 'good' : val > 0.8 ? 'warn' : 'bad';
  if (key === 'clipfrac') return val < 0.2 ? 'good' : val < 0.5 ? 'warn' : 'bad';
  if (key === 'approx_kl') return val < 0.05 ? 'good' : val < 0.2 ? 'warn' : 'bad';
  return '';
}

async function refresh() {
  try {
    let r = await fetch('/stats.json?t='+Date.now());
    if (r.ok) {
      let d = await r.json();
      lastStatsTime = Date.now();
      let label = d.run_name || d.env_name || 'training';
      let cov = d.heatmap_coverage_pct !== undefined ? ' | coverage ' + d.heatmap_coverage_pct.toFixed(2) + '%' : '';
      document.getElementById('update-time').textContent =
        label + ' | step ' + fmt(d.steps) + ' | ' + fmt(d.sps) + ' SPS | epoch ' + fmt(d.epoch) + ' | ' + fmtTime(d.uptime) + cov;

      let html = '';

      // Progression
      html += '<div class="stat-group"><h3>Progression</h3>';
      for (let [k,v] of [['badges_earned',d.badges_earned],['events_completed',d.events_completed],
          ['levels_gained',d.levels_gained],['seen_pokemon',d.seen_pokemon],
          ['caught_pokemon',d.caught_pokemon],['hm_count',d.hm_count],
          ['unique_moves',d.unique_moves],['pokecenter_heals',d.pokecenter_heals]]) {
        if (v !== undefined) html += '<div class="stat-row"><span class="label">'+k+'</span><span class="value">'+fmt(v)+'</span></div>';
      }
      html += '</div>';

      // Exploration
      html += '<div class="stat-group"><h3>Exploration</h3>';
      for (let [k,v] of [['unique_tiles',d.unique_tiles],['unique_maps',d.unique_maps],
          ['warps_taken',d.warps_taken],['battles_won',d.battles_won]]) {
        if (v !== undefined) html += '<div class="stat-row"><span class="label">'+k+'</span><span class="value">'+fmt(v)+'</span></div>';
      }
      html += '<div class="stat-row"><span class="label">episode_return</span><span class="value">'+fmt(d.episode_return)+'</span></div>';
      html += '<div class="stat-row"><span class="label">global_tiles</span><span class="value good">'+fmt(d.global_tiles)+'</span></div>';
      if (d.heatmap_coverage_pct !== undefined) {
        let cpct = d.heatmap_coverage_pct;
        let cc = cpct > 5 ? 'good' : cpct > 2 ? 'warn' : '';
        html += '<div class="stat-row"><span class="label">coverage %</span><span class="value '+cc+'" style="font-size:18px">'+cpct.toFixed(2)+'%</span></div>';
      }
      html += '</div>';

      // Losses
      html += '<div class="stat-group"><h3>Losses</h3>';
      for (let [k,v] of [['pg_loss',d.pg_loss],['vf_loss',d.vf_loss],['entropy',d.entropy],
          ['clipfrac',d.clipfrac],['approx_kl',d.approx_kl],['total_loss',d.total_loss]]) {
        if (v !== undefined) {
          let c = classify(k,v);
          html += '<div class="stat-row"><span class="label">'+k+'</span><span class="value '+c+'">'+fmt(v)+'</span></div>';
        }
      }
      html += '</div>';

      // Maps visited
      if (d.maps && d.maps.length > 0) {
        html += '<div class="stat-group"><h3>Maps ('+d.maps.length+' visited)</h3>';
        html += '<table class="map-table"><tr><th>Map</th><th>Tiles</th><th>Visits</th></tr>';
        for (let m of d.maps.slice(0, 20)) {
          html += '<tr><td>'+m[1]+'</td><td>'+m[2]+'</td><td>'+fmt(m[3])+'</td></tr>';
        }
        html += '</table></div>';
      }

      // Append GIFs after stats
      html += '<div class="stat-group" style="margin-top:16px"><h3>Eval Trajectory</h3>';
      html += '<img src="/eval_trajectory.gif?t='+Date.now()+'" style="max-width:100%;image-rendering:pixelated;border:1px solid #333" onerror="this.style.display=\\'none\\'">';
      html += '</div>';
      html += '<div class="stat-group" style="margin-top:16px"><h3>Agent Obs (live)</h3>';
      html += '<img src="/obs_gif?t='+Date.now()+'" style="max-width:100%;image-rendering:pixelated;border:1px solid #333" onerror="this.style.display=\\'none\\'">';
      html += '</div>';

      document.getElementById('stats-panel').innerHTML = html;
    }
  } catch(e) {}

  // Heatmap image
  try {
    let img = document.querySelector('#heatmap-panel img');
    if (!img) {
      img = document.createElement('img');
      document.getElementById('heatmap-panel').innerHTML = '';
      document.getElementById('heatmap-panel').appendChild(img);
    }
    img.src = '/heatmap.png?t='+Date.now();
  } catch(e) {}

  // Zoom image
  try {
    let zp = document.getElementById('zoom-panel');
    let zimg = zp.querySelector('img');
    if (!zimg) {
      zimg = document.createElement('img');
      zp.innerHTML = '<h3>Pallet → Pewter</h3>';
      zp.appendChild(zimg);
    }
    zimg.src = '/heatmap_zoom.png?t='+Date.now();
  } catch(e) {}
}

setInterval(refresh, 5000);
refresh();
</script>
</body>
</html>"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path.startswith('/heatmap.png'):
            fpath = os.path.join(DASH_DIR, 'heatmap.png')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/heatmap_zoom.png'):
            fpath = os.path.join(DASH_DIR, 'heatmap_zoom.png')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/eval'):
            fpath = os.path.join(DASH_DIR, 'eval.html')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/agent_obs'):
            fpath = os.path.join(DASH_DIR, 'agent_obs.html')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/stats.json'):
            fpath = os.path.join(DASH_DIR, 'stats.json')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/eval_trajectory.gif'):
            fpath = os.path.join(DASH_DIR, 'eval_trajectory.gif')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/gif')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/obs_gif'):
            fpath = os.path.join(DASH_DIR, 'obs_recording.gif')
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/gif')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence request logs

if __name__ == '__main__':
    os.makedirs(DASH_DIR, exist_ok=True)
    server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Dashboard serving at http://0.0.0.0:{PORT}/')
    server.serve_forever()
