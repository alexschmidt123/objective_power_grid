"""Compact execution-host metadata for newly written run configurations."""
import os
from pathlib import Path
import platform
import socket
import subprocess


def hardware_info():
    name=socket.gethostname()
    cluster=os.environ.get('SLURM_CLUSTER_NAME') or os.environ.get('BOED_CLUSTER')
    site=os.environ.get('BOED_EXECUTION_SITE')
    if site not in {'hprc','labpc'}:
        if cluster in {'faster','grace','terra'} or '.hprc.tamu.edu' in name:
            site='hprc'
        elif name.split('.')[0] in {'labpc','ecen-80r5x04'}:
            site='labpc'
        else:
            site='unknown'
    if site!='hprc':cluster=None
    cpu=platform.processor() or None
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu=line.split(':',1)[1].strip();break
    except OSError:pass
    gpu=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=5,check=True)
        names=list(dict.fromkeys(x.strip() for x in result.stdout.splitlines() if x.strip()))
        gpu=', '.join(names) or None
    except (OSError,subprocess.SubprocessError):pass
    return {'execution_site':site,'cluster':cluster,'computer_name':name,
            'cpu_model':cpu,'gpu_model':gpu}
