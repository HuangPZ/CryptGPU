python ./distributed_launcher.py \
    --prepare_cmd="source /users/lisali12/miniconda3/etc/profile.d/conda.sh && conda activate cryptgpu" \
    --aux_files=benchmark.py,network.py \
    --cuda_visible_devices=""\
    launcher.py \
