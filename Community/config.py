dataset = 'FUNC_rand'
bert_path = '/root/data/hatebert_en/'
is_sparse = True   # saprce or not
lr = 0.005  # learning rate
hidden = 300
epoch = 600
cuda_id = 0
nclass = 2
max_sen_len = 50
n_fold = 10
model_name = 'RSGCN'
heads = 1
sim = 0.3
drop = 0.1
fusion = 'Att'   # Sum Max Cat
L = 2