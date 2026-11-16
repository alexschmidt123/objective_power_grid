import threading,signal,subprocess,resource,gc
from common import *
from train import run_pair
from evaluation import run_evaluation

def telemetry(stop):
    previous=time.monotonic();cpu0=time.process_time()
    while not stop.wait(CFG['telemetry_seconds']):
        try:
            now=time.monotonic();cpu=time.process_time()
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
            row=dict(time=time.time(),CPU_percent_one_core=100*(cpu-cpu0)/(now-previous),process_CPU_seconds=cpu,
                     RSS_bytes=int(pathlib.Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                     GPU_percent=float(gpu[0]),GPU_memory_MiB=float(gpu[1]),GPU_power_W=float(gpu[2]))
            previous=now;cpu0=cpu
            with (O/'TRAINING_LOGS/resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        except Exception as exc:
            with (O/'TRAINING_LOGS/telemetry_errors.log').open('a') as f:f.write(str(exc)+'\n')

def main():
    assert not (O/'evidence/EXPERIMENT_STARTED.json').exists()
    manifest=json.loads((O/'PRETRAIN_MANIFEST.json').read_text())
    for name,h in manifest['files'].items():assert sha(O/name)==h,name
    refs=json.loads((O/'evidence/PROTECTED_INPUTS_BEFORE.json').read_text())
    for name,h in refs.items():assert sha(name)==h,name
    original.setup()
    while True:
        try:leases=original.acquire();break
        except BlockingIOError:
            put(O/'QUEUE_STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',pid=os.getpid(),time=time.time()));time.sleep(30)
    start=time.monotonic();stop=threading.Event();threading.Thread(target=telemetry,args=(stop,),daemon=True).start()
    put(O/'evidence/EXPERIMENT_STARTED.json',dict(pid=os.getpid(),time=time.time(),git_commit=manifest['git_commit'],GPU=torch.cuda.get_device_name()))
    def watchdog():
        if not stop.wait(CFG['hard_wall_seconds']-300):os.kill(os.getpid(),signal.SIGTERM)
        if not stop.wait(300):os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start();status='INCOMPLETE'
    try:
        if run_pair(start):
            run_evaluation();status='COMPLETE' if json.loads((O/'evaluation/COMPLETION.json').read_text())['status']=='COMPLETE' else 'EVALUATION_INCOMPLETE'
    except Exception as exc:
        import traceback
        traceback.print_exc();status='FAILED';put(O/'evidence/ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
    finally:
        stop.set();put(O/'evidence/EXPERIMENT_COMPLETION.json',dict(status=status,seconds=time.monotonic()-start,no_extension=True))
        try:
            for name,h in refs.items():assert sha(name)==h,name
            put(O/'evidence/PRESERVATION_VERIFIED.json',dict(status='PASS',files=len(refs)))
        except Exception as exc:
            status='PRESERVATION_FAILED';put(O/'evidence/PRESERVATION_ERROR.json',dict(error=str(exc)))
            put(O/'evidence/EXPERIMENT_COMPLETION.json',dict(status=status,seconds=time.monotonic()-start,no_extension=True))
        from reports import write
        write(O);print('CER_EXPERIMENT_END',status,flush=True)

if __name__=='__main__':main()
