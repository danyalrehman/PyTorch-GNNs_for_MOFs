jobPath = "C:/Users/opbir/CGCNN/"
jobName = "trainCGCNN_test"
dataPath = "C:/Users/opbir/CGCNN/"
disable_checkpt = "store_true"
fp64 = "store_true"
run_dataProcess = False

optimizer = "Adamax"
lr = 1e-2
weight_decay = 1e-2
scheduler_gamma = 0.96

num_epoch = 10
train_batchSize = 256
val_batchSize = 256
test_batchSize = 256
train = True
test = True