SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:\
$SITE_PACKAGES/nvidia/cuda_runtime/lib:\
$SITE_PACKAGES/nvidia/cudnn/lib:\
$SITE_PACKAGES/nvidia/cufft/lib:\
$SITE_PACKAGES/nvidia/curand/lib:\
$SITE_PACKAGES/nvidia/cusolver/lib:\
$SITE_PACKAGES/nvidia/cusparse/lib:\
$SITE_PACKAGES/nvidia/nvtx/lib:\
$LD_LIBRARY_PATH

echo "ORT CUDA library path configured."
