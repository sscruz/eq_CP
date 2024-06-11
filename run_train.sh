#!/bin/bash
# Script para activar el entorno de conda y ejecutar el train

# Configurar el script para salir inmediatamente si algún comando falla
set -e

# Activa el entorno de conda
conda init
conda activate pytorch_1.13

# Cambiar al directorio donde está el script train.py
Cd /nfs/fanae/user/uo278174/[TFG]/eq_CP
# Define el comando
COMMAND="python train.py --name ttbar --data-path /lustrefs/hdd_pool_dir/eq_ntuples/minitrees/ttbar_dil/ --analysis ttbar --batch 1000 --lr 1e-5 --device cpu"

# Ejecuta el comando
$COMMAND

