"""Owned loopback tunnel or append-safe evidence mirror. No remote command execution."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,re,select,socketserver,stat,threading,time,uuid
import paramiko

REMOTE='/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r2'
NAS=Path(r'R:\pufferlib\rek-evidence\2026-09-27\rek-native-clone-r1\live-session-r1')
HERE=Path(__file__).resolve().parent
APPEND={'session.jsonl','resource.jsonl','worker.stderr.log','server.stdout.log','server.stderr.log'}
IMMUTABLE={'identity.json','worker.json','server.json','league.json'}
SNAPSHOT={'recorder-health.json'}
CHUNK=1024*1024;GUARD=65536
CHECKPOINT_RECORDS=32;CHECKPOINT_SECONDS=2.0
FRAME_BATCH=32;FRAME_SECONDS=1.0
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def deadline_for(args):
    return None if getattr(args,'until_stop',False) else time.monotonic()+args.minutes*60
def expired(deadline):return deadline is not None and time.monotonic()>=deadline
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for attempt in range(10):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.02*(attempt+1))
def scoped(root,relative):
    p=PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in ('.','..') or '\\' in x or ':' in x for x in p.parts):raise ValueError('Invalid path')
    out=root.resolve().joinpath(*p.parts).resolve()
    if not out.is_relative_to(root.resolve()) or out==root.resolve():raise ValueError('Path escape')
    return out
def connect():
    config=paramiko.SSHConfig.from_path(str(Path.home()/'.ssh/config')).lookup('dgx_spark')
    if config.get('proxycommand') or config.get('proxyjump'):raise RuntimeError('This saved alias now requires a reviewed proxy setup')
    client=paramiko.SSHClient();client.load_system_host_keys()
    hosts=config.get('userknownhostsfile',[])
    if isinstance(hosts,str):hosts=[hosts]
    for name in hosts:
        p=Path(name).expanduser()
        if p.exists():client.load_host_keys(str(p))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(config.get('hostname','dgx_spark'),port=int(config.get('port',22)),username=config.get('user'),
        key_filename=config.get('identityfile'),timeout=7,banner_timeout=7,auth_timeout=7,allow_agent=True,look_for_keys=True)
    client.get_transport().set_keepalive(10)
    return client
class OwnedLock:
    def __init__(self,path):self.path=path
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.f=self.path.open('a+b');self.f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                if self.path.stat().st_size==0:self.f.write(b'0');self.f.flush();self.f.seek(0)
                msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.f.close();raise
        return self
    def __exit__(self,*args):self.f.close()
def read_exact(f,n):
    out=bytearray()
    while len(out)<n:
        b=f.read(n-len(out))
        if not b:raise IOError('Source ended before captured size')
        out.extend(b)
    return bytes(out)
def prefix_guard(source,target,offset):
    if not offset:return
    n=min(GUARD,offset);source.seek(offset-n)
    with target.open('rb') as f:f.seek(offset-n);old=read_exact(f,n)
    if read_exact(source,n)!=old:raise IOError('Append prefix changed; refusing mixed-session data')
def append_file(source,target,size,stop):
    target.parent.mkdir(parents=True,exist_ok=True);offset=target.stat().st_size if target.exists() else 0
    if offset>size:raise IOError('Source truncated; preserving existing prefix')
    prefix_guard(source,target,offset);source.seek(offset)
    added=hashlib.sha256();start=offset;original_offset=offset
    with target.open('ab') as dest:
        while offset<size:
            if stop():raise InterruptedError('Collection stop')
            data=read_exact(source,min(CHUNK,size-offset));dest.write(data);added.update(data);offset+=len(data)
        dest.flush();os.fsync(dest.fileno())
    check=hashlib.sha256()
    with target.open('rb') as f:
        f.seek(start)
        while start<size:
            data=read_exact(f,min(CHUNK,size-start));check.update(data);start+=len(data)
    if check.digest()!=added.digest():raise IOError('Append readback hash mismatch')
    return {'bytes':size,'added_bytes':size-original_offset,
        'appended_chunk_sha256':check.hexdigest(),'active_prefix':True,'tail_guard_bytes':min(GUARD,size)}
def immutable_file(sftp,remote,local,nas,stop):
    local.parent.mkdir(parents=True,exist_ok=True);nas.parent.mkdir(parents=True,exist_ok=True)
    stage=local.with_name(local.name+'.incoming-'+uuid.uuid4().hex)
    with sftp.open(remote,'rb') as source,stage.open('xb') as out:
        before=source.stat();left=before.st_size;h=hashlib.sha256()
        while left:
            if stop():raise InterruptedError('Collection stop')
            data=read_exact(source,min(CHUNK,left));out.write(data);h.update(data);left-=len(data)
        out.flush();os.fsync(out.fileno());after=source.stat()
    if (before.st_size,before.st_mtime)!=(after.st_size,after.st_mtime):raise IOError('Immutable source changed during copy')
    digest=h.hexdigest()
    if local.suffix=='.png':
        with stage.open('rb') as f:
            if f.read(8)!=b'\x89PNG\r\n\x1a\n':raise IOError('PNG header incomplete')
            f.seek(-12,2)
            if f.read()!=b'\x00\x00\x00\x00IEND\xaeB`\x82':raise IOError('PNG footer incomplete')
    if local.suffix=='.json':json.loads(stage.read_text(encoding='utf-8-sig'))
    for destination in (local,nas):
        if destination.exists():
            if destination.stat().st_size!=before.st_size or sha(destination)!=digest:raise IOError('Immutable destination differs; preserving both')
            continue
        tmp=destination.with_name(destination.name+'.incoming-'+uuid.uuid4().hex)
        with stage.open('rb') as src,tmp.open('xb') as out:
            while data:=src.read(CHUNK):
                if stop():raise InterruptedError('Collection stop')
                out.write(data)
            out.flush();os.fsync(out.fileno())
        if sha(tmp)!=digest:raise IOError('Immutable readback hash mismatch')
        if destination.exists():raise IOError('Destination appeared during copy')
        tmp.rename(destination)
    stage.unlink()
    return {'bytes':before.st_size,'sha256':digest,'source_mtime':before.st_mtime,'source_local_nas_sha_equal':True,'active_prefix':False}
def frame_listing(sftp):
    try:
        if not stat.S_ISDIR(sftp.lstat(REMOTE+'/frames').st_mode):raise IOError('Unexpected frame directory')
        frames=sftp.listdir_attr(REMOTE+'/frames')
    except FileNotFoundError:frames=[]
    files=[]
    for a in sorted(frames,key=lambda x:x.filename):
        if not stat.S_ISREG(a.st_mode) or not a.filename.endswith('.png') or not a.filename[:-4].isdigit():raise IOError('Unexpected frame entry')
        files.append(('frames/'+a.filename,a))
    return files

def listing(sftp,include_frames=True):
    files=[]
    # Refresh health and active prefixes before enumerating/copying PNG backlog.
    for name in sorted(APPEND|IMMUTABLE|SNAPSHOT,key=lambda n:(n not in SNAPSHOT,n not in APPEND,n)):
        try:a=sftp.lstat(REMOTE+'/'+name)
        except FileNotFoundError:continue
        if not stat.S_ISREG(a.st_mode):raise IOError('Unexpected nonregular source: '+name)
        files.append((name,a))
    if include_frames:files.extend(frame_listing(sftp))
    return files

class InventoryCheckpoint:
    def __init__(self,local,nas,identity,files):
        self.local=local;self.nas=nas;self.identity=identity;self.files=files
        self.dirty=0;self.last=time.monotonic()
    def record(self,relative,result):
        self.files[relative]=result;self.dirty+=1
    def flush(self,force=False):
        if not self.dirty:return False
        if not force and self.dirty<CHECKPOINT_RECORDS and time.monotonic()-self.last<CHECKPOINT_SECONDS:return False
        value={'identity':self.identity,'files':self.files,'checkpoint_utc':utc()}
        atomic(self.local,value);atomic(self.nas,value)
        # A failed second write keeps the batch dirty and safe to retry.
        self.dirty=0;self.last=time.monotonic();return True
def initialize(local,nas,resume):
    if resume:
        a=json.loads((local/'MIRROR-IDENTITY.json').read_text());b=json.loads((nas/'MIRROR-IDENTITY.json').read_text())
        if a!=b or a['source']!=REMOTE or a['nas']!=str(nas):raise RuntimeError('Resume ownership identity mismatch')
        return a
    if local.exists() or nas.exists():raise FileExistsError('Fresh local spool and NAS subtree required')
    local.mkdir(parents=True);nas.mkdir(parents=True)
    data={'schema':'rek.native_clone.passive_mirror.v1','id':uuid.uuid4().hex,'utc':utc(),'source':REMOTE,'nas':str(nas),'local':str(local),'script_sha256':sha(Path(__file__))}
    for p in (local,nas):atomic(p/'MIRROR-IDENTITY.json',data)
    return data
def mirror(args):
    local=args.local/'spool';nas=args.nas;stop=lambda: (args.local/'STOP').exists() or expired(deadline)
    deadline=deadline_for(args);identity=initialize(local,nas,args.resume)
    status_path=args.local/'mirror-status.json';ledger_path=local/'inventory.json'
    ledger=json.loads(ledger_path.read_text()) if args.resume and ledger_path.exists() else {'identity':identity['id'],'files':{}}
    if ledger.get('identity')!=identity['id'] or not isinstance(ledger.get('files'),dict):raise RuntimeError('Resume inventory identity mismatch')
    files=ledger['files'];inventory=InventoryCheckpoint(ledger_path,nas/'inventory.json',identity['id'],files)
    # A checkpoint is an index, not proof that files survived a crash unchanged.
    # Recheck each immutable source/local/NAS once after a process starts.
    verified=set()
    client=None;errors=[];last_success=None;source_latest=None;reason='deadline';source_health=None
    def publish_status(phase,**extra):
        status={'utc':utc(),'pid':os.getpid(),'phase':phase,'source':REMOTE,'nas':str(nas),'source_latest_mtime':source_latest,
            'source_age_seconds':time.time()-source_latest if source_latest else None,'last_success_utc':last_success,
            'errors':errors[-10:] if phase=='retrying' else [],'files_verified':len(files),'files_verified_this_process':len(verified),
            'files_verified_scope':'inventory entries, including prior checkpoints; process count is separate','active_append_files_are_prefixes':True,
            'source_recorder_health':source_health,**extra}
        atomic(status_path,status);atomic(nas/'mirror-status.json',status)
    try:
        while not stop():
            frame_backlog=False
            try:
                if client is None:client=connect()
                sftp=client.open_sftp();sftp.get_channel().settimeout(10)
                with sftp:
                    for frame_phase in (False,True):
                        if stop():break
                        entries=frame_listing(sftp) if frame_phase else listing(sftp,include_frames=False)
                        frame_started=time.monotonic();frame_count=0
                        for relative,a in entries:
                            if stop():break
                            source_latest=max(source_latest or 0,a.st_mtime)
                            prior=files.get(relative);signature=[a.st_size,a.st_mtime]
                            if relative in verified and prior and prior.get('source_signature')==signature and relative not in APPEND:continue
                            if frame_phase and (frame_count>=FRAME_BATCH or (frame_count and time.monotonic()-frame_started>=FRAME_SECONDS)):
                                frame_backlog=True;break
                            atomic(status_path,{'utc':utc(),'pid':os.getpid(),'phase':'copying','current_file':relative,'source_latest_mtime':source_latest,'last_success_utc':last_success,'errors':errors[-10:]})
                            src=REMOTE+'/'+relative
                            if relative in APPEND:
                                with sftp.open(src,'rb') as source:
                                    if source.stat().st_size<a.st_size:raise IOError('Source shrank before append')
                                    lp=scoped(local,relative);np=scoped(nas,relative)
                                    append_file(source,lp,a.st_size,stop)
                                    with lp.open('rb') as copied:result=append_file(copied,np,a.st_size,stop)
                                result['verification']='source-to-spool-to-NAS append bytes readback; prior last64KiB checked'
                            else:
                                dest=relative
                                if relative in SNAPSHOT:dest='snapshots/'+relative[:-5]+'/'+str(a.st_mtime)+'-'+str(a.st_size)+'.json'
                                result=immutable_file(sftp,src,scoped(local,dest),scoped(nas,dest),stop);result['destination']=dest
                                if relative=='recorder-health.json':source_health=json.loads(scoped(local,dest).read_text())
                            result.update({'source_signature':signature,'verified_utc':utc()})
                            inventory.record(relative,result);verified.add(relative);inventory.flush()
                            if frame_phase:frame_count+=1
                        if not frame_phase:publish_status('watching',frame_backlog_pending=True)
                    inventory.flush(force=True)
                    last_success=utc()
                    publish_status('watching',frame_backlog_pending=frame_backlog)
            except InterruptedError:break
            except Exception as e:
                errors.append({'utc':utc(),'error':type(e).__name__+': '+str(e)})
                try:inventory.flush(force=True)
                except Exception as checkpoint_error:errors.append({'utc':utc(),'error':'Error checkpoint failed: '+str(checkpoint_error)})
                atomic(status_path,{'utc':utc(),'pid':os.getpid(),'phase':'retrying','last_success_utc':last_success,'errors':errors[-10:]})
                try:publish_status('retrying')
                except Exception:pass # Local retry status already preserves the error when NAS is unavailable.
                if client:client.close();client=None
                frame_backlog=False
            for _ in range(0 if frame_backlog else 30):
                if stop():break
                time.sleep(.1)
        if (args.local/'STOP').exists():reason='STOP'
    finally:
        if client:client.close()
        checkpoint_errors=[]
        try:inventory.flush(force=True)
        except Exception as e:checkpoint_errors.append({'utc':utc(),'error':'Final inventory checkpoint failed: '+str(e)})
        final={'utc':utc(),'pid':os.getpid(),'phase':'stopped','reason':reason,'last_success_utc':last_success,
            'errors':checkpoint_errors,'historical_errors':errors[-10:]}
        atomic(status_path,final)
        try:atomic(nas/'mirror-status.json',final)
        except Exception as e:
            final['errors'].append({'utc':utc(),'error':'Final NAS status publication failed: '+str(e)})
            atomic(status_path,final)

def tunnel(args):
    deadline=deadline_for(args);halt=threading.Event()
    stop=lambda:halt.is_set() or (args.local/'STOP').exists() or expired(deadline)
    client=connect();transport=client.get_transport();counts={'connections':0,'client_bytes':0,'remote_bytes':0};errors=[]
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel=None;counts['connections']+=1
            try:
                channel=transport.open_channel('direct-tcpip',('127.0.0.1',args.port),self.request.getpeername(),timeout=7)
                channel.settimeout(2);self.request.settimeout(2)
                while not stop():
                    ready,_,_=select.select([self.request,channel],[],[],.5)
                    for src in ready:
                        data=src.recv(65536)
                        if not data:return
                        if src is self.request:channel.sendall(data);counts['client_bytes']+=len(data)
                        else:self.request.sendall(data);counts['remote_bytes']+=len(data)
            except Exception as e:errors.append({'utc':utc(),'error':str(e)})
            finally:
                if channel:channel.close()
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=False;daemon_threads=True
    server=None
    try:
        server=Server(('127.0.0.1',args.port),Handler);server.timeout=.5;written=0
        while not stop():
            server.handle_request()
            if not transport.is_active():raise ConnectionError('SSH transport disconnected')
            if time.monotonic()-written>=2:
                endpoint='127.0.0.1:'+str(args.port)
                atomic(args.local/'tunnel-status.json',{'utc':utc(),'pid':os.getpid(),'phase':'listening','local':endpoint,'remote':endpoint,'counts':counts,'errors':errors[-10:]});written=time.monotonic()
    finally:
        halt.set()
        if server:server.server_close()
        client.close();atomic(args.local/'tunnel-status.json',{'utc':utc(),'pid':os.getpid(),'phase':'stopped','reason':'STOP' if (args.local/'STOP').exists() else 'deadline_or_error','counts':counts,'errors':errors[-10:]})
def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['tunnel','mirror']);p.add_argument('--local',type=Path,default=HERE/'active');p.add_argument('--nas',type=Path,default=NAS)
    p.add_argument('--port',type=int,default=18771,help='Matching local and remote loopback port')
    lifetime=p.add_mutually_exclusive_group()
    lifetime.add_argument('--minutes',type=float,help='Stop after this many minutes, at most240; default240')
    lifetime.add_argument('--until-stop',action='store_true',help='No timed expiry; stop using the shared STOP marker')
    p.add_argument('--resume',action='store_true');p.add_argument('--remote',default=REMOTE);args=p.parse_args(argv)
    if not 1024<=args.port<=65535:raise ValueError('Loopback port must be1024..65535')
    if args.minutes is None and not args.until_stop:args.minutes=240
    if not args.until_stop and not 0<args.minutes<=240:raise ValueError('Bounded duration required, at most240minutes')
    if not re.fullmatch(r'/home/spark-advantage/rek-training/rek-native-clone-20260927-r1/run-r[0-9]+',args.remote):raise ValueError('Remote must be a named isolated native clone run')
    return args
def main():
    global REMOTE
    args=parse_args()
    REMOTE=args.remote
    args.local.mkdir(parents=True,exist_ok=True)
    if (args.local/'STOP').exists():raise RuntimeError('STOP exists; preserving stop state')
    with OwnedLock(args.local/(args.mode+'.lock')):(mirror if args.mode=='mirror' else tunnel)(args)
if __name__=='__main__':main()
