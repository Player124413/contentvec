#!/bin/bash
if [ -d "/usr/lib/wsl" ]; then
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
fi
CONDA_ROOT=/home/$(whoami)/miniconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate contentvec
wget https://huggingface.co/ShiromiyaGamer/dependencias/resolve/main/checkpoint_best_100.pt

mkdir feature
mkdir feature/lab

python3 -m pip install npy_append_array scikit-learn joblib resemblyzer pyreaper

python3 fairseq/examples/wav2vec/wav2vec_manifest.py dataset --dest feature --valid-percent 0.01

cp feature/train.tsv feature/lab/train.tsv
cp feature/valid.tsv feature/lab/valid.tsv
rm -rf fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py
cp dump_hubert_feature.py fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py

tsv_dir="feature"
split="train"
ckpt_path="checkpoint_best_100.pt"
layer=12
nshard=1
rank=0
feat_dir="feature"
km_path="feature/${split}.km"
lab_dir="feature/lab"
n_clusters=100

python speaker.py

python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py $tsv_dir $split $ckpt_path $layer $nshard $rank $feat_dir

split="valid"

python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py $tsv_dir $split $ckpt_path $layer $nshard $rank $feat_dir

split="train"

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py $feat_dir $split $nshard $km_path $n_clusters --percent 0.1

split="valid"

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py $feat_dir $split $nshard $km_path $n_clusters --percent 0.1

split="train"

python fairseq/examples/hubert/simple_kmeans/dump_km_label.py $feat_dir $split $km_path $nshard $rank $lab_dir

split="valid"

python fairseq/examples/hubert/simple_kmeans/dump_km_label.py $feat_dir $split $km_path $nshard $rank $lab_dir

split="train"

for rank in $(seq 0 $((nshard - 1))); do
    cat "${lab_dir}/${split}_0_${nshard}.km"
done > "${lab_dir}/${split}.km"

split="valid"

for rank in $(seq 0 $((nshard - 1))); do
    cat "${lab_dir}/${split}_0_${nshard}.km"
done > "${lab_dir}/${split}.km"

for x in $(seq 0 $((n_clusters - 1))); do
   sudo bash -c "echo '$x 1' >> $lab_dir/dict.km.txt"
done

expdir=./tmp

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE_JOB=\$SLURM_NTASKS
RANK_NODE=\$SLURM_NODEID
PROC_PER_NODE=4
MASTER_ADDR_JOB=\$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"
DDP_BACKEND=c10d

rm -rf fairseq/fairseq/data/audio/contentvec_dataset.py
cp contentvec_dataset.py fairseq/fairseq/data/audio/contentvec_dataset.py

HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py  \
    --config-dir ./contentvec/config/contentvec \
    --config-name contentvec \
    hydra.run.dir=$expdir \
    task.data=/home/$(whoami)/contentvec-main/feature/lab \
    task.label_dir=/home/$(whoami)/contentvec-main/feature/lab \
    task.labels=["km"] \
    task.spk2info=/home/$(whoami)/contentvec-main/feature/lab/spk2info.dict \
    task.crop=true \
    dataset.train_subset=train \
    dataset.valid_subset=valid \
    dataset.num_workers=10 \
    dataset.max_tokens=1230000 \
    checkpoint.keep_best_checkpoints=10 \
    criterion.loss_weights=[10,1e-5] \
    model.label_rate=50 \
    model.encoder_layers_1=3 \
    model.logit_temp_ctr=0.1 \
    model.ctr_layers=[-6] \
    model.extractor_mode="default" \
    optimization.update_freq=[1] \
    optimization.max_update=100000 \
    lr_scheduler.warmup_updates=8000 \
    2>&1
