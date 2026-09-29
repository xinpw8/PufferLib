"""Read-only guards before starting an isolated paused corrected viewer."""
from pathlib import Path
import hashlib,json,subprocess,sys,urllib.request
EXPECTED={1092025:125031755,1092037:125031762,3297624:132749534,3297634:132749542,3297635:132749542}

def main():
    for pid,start in EXPECTED.items():
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        assert int(fields[19])==start,'Existing process identity changed'
    observed={}
    for port in [18771,18772]:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/snapshot',timeout=2) as response:value=json.load(response)
        assert value['ok'] and value['paused'],'Existing human viewer resumed; no launch'
        observed[port]={'paused':True,'tick':value['tick']}
    launch=Path(__file__).with_name('launch_viewer.py')
    assert hashlib.sha256(launch.read_bytes()).hexdigest()=='0d411a7ae5e7a66d103d85a568b3d3b5ff4b356c62b60409afc49819bcbae570'
    print(json.dumps({'existing_viewers_observed_only':observed}),flush=True)
    subprocess.run([sys.executable,str(launch),*sys.argv[1:]],check=True)

if __name__=='__main__':main()
