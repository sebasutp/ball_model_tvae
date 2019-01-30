
import traj_pred.cvae as cvae
import os
import json

traj_loaders = {'cvae': cvae.load_traj_cvae}

def load_model(path):
    conf = json.load( open(os.path.join(path,'conf.json'), 'r') )
    if not 'model' in conf:
        raise Exception('The keyword "model" must be present in the configuration JSON file')
    model = conf['model']
    if not model in traj_loaders:
        raise Exception('Model {} not recognized'.format(model))
    return traj_loaders[model](path)
