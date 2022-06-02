import subprocess
import time

dir = "NewTest/"
def work(dataset, gpu_id):
    cmd = f"nohup python main.py --dataset {dataset} --path {dir} --device {gpu_id} > {dir}/{dataset}_{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def opt(dataset, gpu_id):
    dir = "FWLOpt"
    cmd = f"nohup python hyperopt.py --dataset {dataset} --device {gpu_id} > {dir}/{dataset}_{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def test(dataset, gpu_id):
    cmd = f"nohup python main.py --test --device {gpu_id} --dataset {dataset} > {dir}/{dataset}_{gpu_id}.test 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def test_local(dataset, gpu_id):
    cmd = f"nohup python main_local.py --test --device {gpu_id} --dataset {dataset} > {dir}/{dataset}_{gpu_id}.test_local 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def reproduce(dg):
    dir = "reproduceHY"
    dataset, gpu_id = dg
    cmd = f"nohup python main.py --reproduce --device {gpu_id} --dataset {dataset} > {dir}/{dataset}_{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def wait():
    while True:
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        load = [int(i) for i in sel]
        for i in range(len(load) // 2):
            if sum(load[2 * i:2 * i + 2]) < 50:
                return i
        time.sleep(30)
        

for ds in ['Celegans','USAir','PB','NS','Ecoli','Router','Power','Yeast','Cora','Citeseer']:
    dev = wait()
    test(ds, dev)
    time.sleep(40)

for ds in ['Celegans','USAir','PB','NS','Ecoli','Router','Power','Yeast','Cora','Citeseer']:
    dev = wait()
    test_local(ds, dev)
    time.sleep(40)