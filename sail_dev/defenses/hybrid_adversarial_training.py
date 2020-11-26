from art.defences.trainer.trainer import Trainer

import torch

import pdb
import numpy as np

from sail_dev.models.feature_scatter_attack_hybrid import Attack_FeaScatter
from sail_dev.models import RawAudioCNN
from hparams import hp

from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

# config for feature scatter
config_feature_scatter = {
    'train': True,
    #'epsilon': 8.0 / 255 * 2,
    'epsilon': 0.01,
    'num_steps': 10,
    #'step_size': 8.0 / 255 * 2,
    'step_size': 0.01 / 5,
    'random_start': True,
    'ls_factor': 0.5,
}

def compute_accuracy(targets, pred):
    return accuracy_score(targets, pred)

def seed_random():
    torch.manual_seed(10101)
    np.random.seed(10101)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HybridAdversarialTraining(Trainer):
    """
    Class performing adversarial training following Madry's Protocol.
    Paper link: https://arxiv.org/abs/1706.06083
    Please keep in mind the limitations of defences. While adversarial training is
    widely regarded as a promising, principled approach to making classifiers more
    robust (see https://arxiv.org/abs/1802.00420), very careful evaluations are
    required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(self, classifier=None):
        self.classifier  = classifier 
        self.classification_model = classifier._model._model
        self.model = self._create_model()
        self.criterion = classifier._loss
        self.optimizer = classifier._optimizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.log_dir = '/work/rperi/gard/scripts/eval_Dec2020/logs/FGSM_defended_HAT10AT/epochs_300_Adam_lr1e-4/'
        self.model_dir = '/work/rperi/gard/scripts/eval_Dec2020/saved_models/FGSM_defended_HAT10AT/epochs_300_Adam_lr1e-4/'

        self._create_dirs([self.log_dir,self.model_dir])
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.log_file = os.path.join(self.log_dir, 'output.log')
        self.idx = 0
        self.ep_idx = 0
        
        seed_random()
    def _create_model(self):
        return Attack_FeaScatter(self.classification_model, config_feature_scatter)

    def _create_dirs(self, dirs):
        for direc in dirs:
            if not os.path.exists(direc):
                os.makedirs(direc)

    def fit(self, x, y, **kwargs):
        #self.trainer.fit(x, y, **kwargs)
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        self.model.train()
        init_lr = self.optimizer.param_groups[0]['lr']

        loss_epoch = []
        for epoch in tqdm(range(nb_epochs)):
            labels_epoch = []
            pred_nat_epoch = []
            pred_adv_epoch = []
            #print("Training Epoch {}/{}".format(epoch, nb_epochs))
            self.ep_idx = epoch
            # update learning rate
            if False:
                if epoch < 60:
                    lr = init_lr
                elif epoch < 90:
                    lr = init_lr * 0.1
                else:
                    lr = init_lr * 0.1 * 0.1

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            num_batches = int(generator.size / generator.batch_size)	
            self.train_size = num_batches
            for batch_idx in tqdm(range(num_batches)):
                self.idx = batch_idx
                inputs, labels = generator.get_batch()

                inputs, labels = torch.from_numpy(inputs).to(self._device), torch.from_numpy(labels).to(self._device)
                outputs, loss_fs, _ = self.model(inputs.detach(), labels, attack=True)

                self.optimizer.zero_grad()
                loss_fs.backward()
                self.optimizer.step()

                # Save loss and accuracies
                self.write_loss(loss_fs.detach().cpu().item())
                labels_epoch.append(labels.detach().cpu().tolist())

                nat_outputs, _ = self.model(inputs.detach(), targets=labels, attack=False)
                
                pred_nat_epoch.append(torch.argmax(nat_outputs.detach().cpu(), axis=1).tolist())
                pred_adv_epoch.append(torch.argmax(outputs.detach().cpu(), axis=1).tolist())

            labels_epoch = [x for y in labels_epoch for x in y]
            pred_nat_epoch = [x for y in pred_nat_epoch for x in y]
            pred_adv_epoch = [x for y in pred_adv_epoch for x in y]
            nat_acc = np.round(compute_accuracy(labels_epoch, pred_nat_epoch)*100,4)
            adv_acc = np.round(compute_accuracy(labels_epoch, pred_adv_epoch)*100,4)

            self.write_acc(nat_acc)
            self.write_acc(adv_acc, type='Adv')

            self.write_log(nat_acc, adv_acc)
            torch.save(self.model.basic_net,os.path.join(self.model_dir,'HAT_saved_model_epoch_{}.pth'.format(epoch)))
        print("Done training")
        self.writer.flush()
        self.writer.close()

        
    def write_loss(self, loss_pred):
        idx = self.idx + self.train_size * self.ep_idx
        self.writer.add_scalar('Loss/',
                               loss_pred, idx)

    def write_acc(self, accuracy, type='Nat'):
        self.writer.add_scalar('Accuracy/{}'.format(type),
                               accuracy, self.ep_idx)
   
    def write_log(self, nat_acc, adv_acc):
        with open(self.log_file, 'a') as o:
            if self.ep_idx == 0:
                o.write("Epoch Natural_accuracy Adversarial_accuracy\n")
            o.write("{} {} {}\n".format(self.ep_idx, nat_acc, adv_acc))
