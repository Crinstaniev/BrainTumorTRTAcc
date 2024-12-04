docker run \
  --gpus all \
  --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /home/crinstaniev/Dev/BrainTumorTRTAcc/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.11-py3 \
  tritonserver \
  --model-repository=/models
