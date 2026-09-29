"""Size cap for one viewer run. Stops only the viewer session it was started for.

The app records every rendered frame without a cap. Every 30 s this sums the run
directory; above the run cap, or when the filesystem passes the disk fraction, it
writes one receipt, stops the resource watcher by its STOP file and terminates the
viewer's own process group (verified by PID and start ticks). Exits with the viewer.
"""
from pathlib import Path
import argparse,datetime,json,os,shutil,signal,time

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def start_ticks(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    return int(fields[19]),int(fields[2]),fields[0]

def alive(pid,ticks):
    try:observed,group,state=start_ticks(pid)
    except (FileNotFoundError,ProcessLookupError):return False
    return observed==ticks and group==pid and state not in ('Z','X')

def group_alive(pid):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            _,group,state=start_ticks(int(entry.name))
            if group==pid and state not in ('Z','X'):return True
        except (FileNotFoundError,ProcessLookupError,IndexError):continue
    return False

def run_bytes(run):
    total=0
    for folder,_,names in os.walk(run):
        for name in names:
            try:total+=os.lstat(os.path.join(folder,name)).st_size
            except FileNotFoundError:pass
    return total

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pid',type=int,required=True);p.add_argument('--start-ticks',type=int,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--cap-bytes',type=int,required=True)
    p.add_argument('--disk-fraction',type=float,required=True);p.add_argument('--interval',type=float,default=30)
    args=p.parse_args()
    assert alive(args.pid,args.start_ticks),'Viewer identity unavailable'
    status=args.run/'disk-cap-status.json'
    while alive(args.pid,args.start_ticks):
        size=run_bytes(args.run);disk=shutil.disk_usage(args.run);fraction=disk.used/disk.total
        record={'utc':utc(),'pid':os.getpid(),'viewer_pid':args.pid,'run_bytes':size,'cap_bytes':args.cap_bytes,
                'disk_used_fraction':round(fraction,4),'disk_cap_fraction':args.disk_fraction}
        tmp=status.with_name(status.name+'.tmp');tmp.write_text(json.dumps(record)+'\n');os.replace(tmp,status)
        if size>args.cap_bytes or fraction>args.disk_fraction:
            with (args.run/'DISK-CAP-TRIPPED.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
            try:(args.run/'STOP_RESOURCE_WATCH').open('x').close()
            except FileExistsError:pass
            if alive(args.pid,args.start_ticks):os.killpg(args.pid,signal.SIGTERM)
            deadline=time.monotonic()+20
            while group_alive(args.pid) and time.monotonic()<deadline:time.sleep(.5)
            if group_alive(args.pid):
                try:os.killpg(args.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            return
        time.sleep(args.interval)

if __name__=='__main__':main()
