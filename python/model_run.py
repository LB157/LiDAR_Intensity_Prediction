import sys
import torch
import torch.utils.data as data

import inten
import otils as ot
import torchutils as tu

if __name__ == '__main__':
    config = ot.io.load_multi_yml(sys.argv[1])
    if 'seed' in config:
        tu.seed_all(config['seed'])
    runner = inten.data.Runner(config)
    trn_dataset = data.DataLoader(inten.data.Dataset(config['train']), **config['train_loader'])
    val_dataset = data.DataLoader(inten.data.Dataset(config['val']), **config['val_loader'])
    if 'scheduler' in config:
        scheduler = inten.utils.scheduler(config['scheduler'], runner.optimizer)
    else:
        scheduler = None
    for i in range(config['epochs']):
        trn_loss = runner(trn_dataset, tu.TorchMode.TRAIN)
        val_loss = runner(val_dataset, tu.TorchMode.EVAL)
        if scheduler is not None:
            scheduler.step(val_loss)
        saveMdelname ='/home/public/liubo/lidar-intensity/model/epoch_'+ str(i) +'_'+ str(trn_loss.item())+'_'+ str(val_loss.item())+'.pt'
        # torch.jit.trace(runner.model)
        # # torch.jit.trace(runner.model).save(runner.model.state_dict(), filename)
        torch.save(runner.model.state_dict(), saveMdelname)
