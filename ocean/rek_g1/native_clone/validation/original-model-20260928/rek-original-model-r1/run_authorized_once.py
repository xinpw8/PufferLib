from pathlib import Path
import datetime,importlib.util,json
root=Path(r'C:\rekagent\work\rek-original-model-20260928-r1')
spec=importlib.util.spec_from_file_location('owned_passive',r'C:\rekagent\work\rek-native-clone-20260927-r1\passive-support-r2\passive.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
client=mod.connect()
command='python3 /home/spark-advantage/rek-training/rek-original-model-20260928-r1/package-r1/isolation/orchestrate.py --start'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
    inp,out,err=client.exec_command(command,timeout=240)
    stdout=out.read();stderr=err.read();code=out.channel.recv_exit_status()
    (root/'execution/09-start.stdout').write_bytes(stdout)
    (root/'execution/09-start.stderr').write_bytes(stderr)
    receipt={'command':command,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code}
    (root/'execution/09-start.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:client.close()
