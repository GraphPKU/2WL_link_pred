#!/bin/bash
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --output=jobout/%j.out
#SBATCH --error=jobout/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --gpus=5
#SBATCH --ntasks=5
#SBATCH --comment="test for KG GNN"
#SBATCH --job-name=EquiNet-test
#SBATCH --nodelist=lambda-hyperplane03
module load anaconda3/2021.11
source ~/.bashrc
source activate wxy
CUDA_VISIBLE_DEVICES=0 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt3 2>&1 &
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt4 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt5 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt6 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt7 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py   > collab.opt3 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py  --nc 15 > collab.15.no1wl_cnaa_sigmoid 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py  --nc 25 > collab.25.no1wl_cnaa_sigmoid 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py  --nc 30 > collab.30.no1wl_cnaa_sigmoid 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python clusterroutine.py  --nc 25 > collab.25 2>&1 &
#sleep 20
#CUDA_VISIBLE_DEVICES=1 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt1 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt2 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt3 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt4 2>&1 &
#CUDA_VISIBLE_DEVICES=5 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt5 2>&1 &
#CUDA_VISIBLE_DEVICES=6 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python oneheadroutine.py   --dataset ogbl-ppa  > ppa.opt6 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec1_5 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec2_5 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec3_5 2>&1 &
#CUDA_VISIBLE_DEVICES=4 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec4_5 2>&1 &
#CUDA_VISIBLE_DEVICES=5 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec5_5 2>&1 &
#CUDA_VISIBLE_DEVICES=6 nohup /scratch/mhzhang/.conda/envs/wxy/bin/python mainhr.py   --dataset FB15k-237  > out/FB15k-237.hrrec6_5 2>&1 &
nvidia-smi
wait
