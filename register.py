import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'ml-1m', 'yelp', 'syn']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'yahoo':
    dataset = dataloader.Loader(path="../data/"+world.dataset+'/processed')

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'lgn-navip': model.LightGCN,
    'lgn-adjnorm': model.LightGCN,
    'lgn-apda': model.LightGCN,
    'lgn-pc': model.LightGCN,
    'lgn-reg': model.LightGCN,
    'lgn-macr': model.LightGCN,
    'ours': model.LightGCN,
}