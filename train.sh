echo `hostname`

#export PYTHONPATH="/export/c06/efrat/k2/build/../k2/python:/export/c06/efrat/k2/build/lib:$PYTHONPATH"
export PATH="/export/c06/efrat/miniconda3/envs/secondPytorch/bin:$PATH"
export PYTHONPATH="/export/c06/efrat/icefall:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=`free-gpu 1`
python ./Training_Procedure.py 