"""Use the established host-key-checked transport for this isolated task only."""
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/home/spark-advantage/rek-training/rek-parity-continuation-20260928-r1'
spec=importlib.util.spec_from_file_location('existing_transport',r'C:\rekagent\work\rek-native-clone-20260927-r1\remote_task.py')
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.LOCAL=ROOT
transport.REMOTE=REMOTE

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='upload-package':
        if len(sys.argv)!=3:raise SystemExit('upload-package requires a prepared directory')
        with transport.connect() as client:
            transport.upload_tree(client,Path(sys.argv[2]).resolve(),REMOTE)
    else:
        transport.main()
