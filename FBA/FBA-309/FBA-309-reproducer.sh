pip install git+https://github.com/rocm/composable_kernel@develop
git clone --recursive https://github.com/pytorch/pytorch
pip3 install pytest
pushd pytorch
TORCH_LOGS=+torch._inductor pytest --capture=tee-sys test/inductor/test_ck_backend.py
popd
