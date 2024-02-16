python ./distributed_launcher.py \
    --prepare_cmd="source /users/lisali12/miniconda3/etc/profile.d/conda.sh && conda activate cryptgpu && export LIBRARY_PATH=/usr/local/cuda/lib64\${LIBRARY_PATH:+:\${LIBRARY_PATH}} && export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" \
    --aux_files=benchmark.py,network.py \
    --cuda_visible_devices="0" \
    launcher.py \
