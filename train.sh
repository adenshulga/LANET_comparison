

# #dunnhumby
# device=0
# data="tcmbn_data/dunnhumby/split_1/"
# batch=8
# n_head=6
# n_layers=1
# d_model=32
# d_inner=16
# d_k=16
# d_v=16
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=0.01
# epoch=100
# log=log.txt


# python train_lanet.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -ber_comps $ber_comps -dropout $dropout -lr $lr -epoch $epoch -log $log

#dunnhumby
# device=0
# data="tcmbn_data/dunnhumby/split_1/"
# batch=32
# n_head=6
# n_layers=1
# d_model=32
# d_inner=16
# d_k=16
# d_v=16
# ber_comps=64
# gau_comps=64
# dropout=0.1
# lr=0.01
# epoch=200
# log=log.txt


# CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -ber_comps $ber_comps -dropout $dropout -lr $lr -epoch $epoch -log $log
# python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_inner $d_inner -d_k $d_k -d_v $d_v -ber_comps $ber_comps -dropout $dropout -lr $lr -epoch $epoch -log $log

export PYTHONPATH=.

python Main.py
