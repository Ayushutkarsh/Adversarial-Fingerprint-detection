import argparse
import os
from solver import Solver
from attacker import Attacker
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['LivDet2015']:
        print('ERROR!! model_type should be selected in LivDet2015')
        print('Your input for model_type was %s'%config.model_type)
        return
    if config.attack_type not in ['DE','FGSM','DeepFool']:
        print('ERROR!! model_type should be selected in DE,FGSM,DeepFool')
        print('Your input for attack_type was %s'%config.attack_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    '''
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = 200
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    #config.lr = lr
    config.num_epochs_decay = decay_epoch
    '''

    print(config)
    
    train_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)
    attack_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=1,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    
    
    
    # Train and sample the images
    if config.attack ==1:
        attacker = Attacker(config, attack_loader)
        attacker.train()
    elif config.attack ==0:
        solver = Solver(config, train_loader, valid_loader, test_loader)
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=227)
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.0)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='LivDet2015', help='LivDet2015')
    parser.add_argument('--objective', type=str, default='classification', help='classification/segmentation')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')
    #Adversarial Attack
    parser.add_argument('--attack', type=int, default=0, help='Set on for adversarial attack')
    parser.add_argument('--attack_type', type=str, default='DE', help='DE/FGSM/DeepFool')
    parser.add_argument('--pixels', default=2000, type=int, help='The no. of pixels to be perturbed.')
    parser.add_argument('--maxiter', default=100, type=int, help='Max iter in the DE algorithm.')
    parser.add_argument('--popsize', default=10, type=int, help='No of adverisal eg in each iter')
    parser.add_argument('--samples', default=100, type=int, help='The no of samples to attack.')
    parser.add_argument('--targeted', action='store_true', help='Set on for targeted attacks.')
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
