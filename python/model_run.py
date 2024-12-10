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
    
    #by nhy, 以下两行代码增加与训练结果
    # weights = torch.load('/home/public/liubo/lidar-intensity/python/15000_kitti_epoch_69_0.005458739586174488_0.005712802521884441.pt')
    # weights = torch.load('/home/public/liubo/lidar-intensity/python/intensity_weights.pt')
    # weights = torch.load('/home/public/liubo/lidar-intensity/python/epoch_50_0.05280787870287895_0.06388607621192932.pt')
    # runner.model.load_state_dict(weights)

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
