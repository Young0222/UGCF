import world
import utils
from world import cprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
import tracemalloc
from register import dataset
from parse import parse_args

args = parse_args()

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_recall = 0
    
    tracemalloc.start(25)
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            
            if args.model in {'mf', 'ours-one'} and results['recall'] > best_recall:
                best_recall = results['recall']
                embed_user = Recmodel.embedding_user.weight.detach().cpu().numpy()
                embed_item = Recmodel.embedding_item.weight.detach().cpu().numpy()
                np.save('lgn_embed_user_'+args.dataset+'.npy', embed_user)
                np.save('lgn_embed_item_'+args.dataset+'.npy', embed_item)

        T1 = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        T2 = time.time()
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    size, peak = tracemalloc.get_traced_memory()
    print("Total time: ", (T2-T1)*1000, "ms")
    print('memory blocks:{:>10.4f} KiB'.format(peak / 1024))
finally:
    if world.tensorboard:
        w.close()