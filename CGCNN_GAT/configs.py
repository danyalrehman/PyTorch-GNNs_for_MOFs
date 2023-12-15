import torch

jobPath = "C:/Users/opbir/CGCNN/cgcnn_att_v6/"
jobName = "trainCGCNN_test"
dataPath = "C:/Users/opbir/CGCNN/cgcnn_att_v6/"
disable_checkpt = "store_true"
fp64 = "store_true"
run_dataProcess = False

optimizer = "adam"
lr = 5e-3
weight_decay = 1e-1
scheduler_gamma = 0.96

num_epoch = 150
train_batchSize = 128
val_batchSize = train_batchSize
test_batchSize = train_batchSize
train = True
test = True

n_heads = 4
dropout = 0.2

indicies = [13, 16]
indx_str = f"_{indicies[0]}_{indicies[1]}"
pos = torch.tensor(range(indicies[0],indicies[1]))

num = [0, 5394]