import subprocess
import time

dir = "opt"


def work(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python main.py --device {gpu_id} --dataset {dataset} > {dir}/{dataset}_{gpu_id} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def test(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python main.py --test --device {gpu_id} --dataset {dataset} > {dir}/{dataset}_{gpu_id}.test 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

dir = "reproduceHY"
def reproduce(dg):
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
        time.sleep(5)


for i, ds in enumerate([
        "Power"
]):
    dev = wait()
    reproduce((ds, dev))
    time.sleep(10)

'''
for i, ds in enumerate([
        "Celegans", "NS", "Power", "Router", "Ecoli", "PB", "USAir", "Yeast",
        "Wikipedia", "Wikipedia", "arxiv"
]):
    dev = wait()
    reproduce((ds, dev))
    time.sleep(10)
'''