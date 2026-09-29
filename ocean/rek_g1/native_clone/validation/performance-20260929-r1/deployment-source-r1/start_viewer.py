"""Explicit root-operated startup. Never called by preparation or a timer."""
import json,subprocess,sys
import deployment_config as c
import launch_viewer as launcher

def command():
    return [sys.executable,str(c.HERE/'launch_viewer.py'),'--app',str(c.APP),'--run',str(c.RUN),
        '--manifest-sha256',c.APP_SHA,'--binary-sha256',c.BINARY_SHA,'--port',str(c.PORT),
        '--guard-viewers',str(c.HERE/'human-viewers.json'),'--cpus',c.CPUS]

def main():
    c.verify_package()
    receipt=json.loads((c.HERE/'PREPARED.json').read_text());c.validate_prepared(receipt)
    launcher.check_preserved_viewers(c.guards())
    assert not (c.RUN/'server-process.json').exists(),'Never start over an existing viewer'
    subprocess.run(command(),check=True)

if __name__=='__main__':main()
