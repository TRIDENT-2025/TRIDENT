import os, torch, copy
from tqdm import tqdm
import utils.scheduler as sc
from utils.functions import count_parameters, save
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.metrics import precision_score,precision_recall_curve, average_precision_score, f1_score, recall_score, confusion_matrix
import time 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, recall_score, f1_score, top_k_accuracy_score
from tqdm import tqdm
import time


def train_uni_track_acc(model, architect, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, 
                                parallel, logger, args, init_acc=0.0, status='_search', modal_num=0):


    best_genotype = None
    best_acc = init_acc
    best_epoch = 0

    best_test_acc = init_acc
    best_test_epoch = 0

    top1_train = []
    top5_train = []
    top1_test = []
    top5_test = []

    failsafe = True
    cont_overloop = 0
    while failsafe:
        for epoch in range(num_epochs):
            
            logger.info('Epoch: {}'.format(epoch))            
            logger.info("EXP: {}".format(args.save) )

            phases = []
            if status == 'search':
                phases = ['train', 'dev']
            else:
                phases = ['train', 'test']
            
            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    if architect is not None:
                        architect.log_learning_rate(logger)
                    model.train()  # Set model to training mode
                    list_preds = [] 
                    list_label = []
                            
                elif phase == 'dev':
                    if status == 'eval':
                        if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
                    model.train()
                    list_preds = [] 
                    list_label = [] 
                             
                else:
                    model.eval()  # Set model to evaluate mode
                    list_preds = [] 
                    list_label = []                    
            
                running_loss = 0.0
                with tqdm(dataloaders[phase]) as t:
                    for data in dataloaders[phase]:                                    
                        image, label = data[modal_num], data[-1]
        
                        image = image.to(device).float()
                        label = label.to(device).float()
                        
                        # print(f"Input shape: {image.shape}")


                        if status == 'search' and (phase == 'dev' or phase == 'test'):
                            architect.step((image), label, logger)
        
                        optimizer.zero_grad()
        
                        with torch.set_grad_enabled(phase == 'train' or (phase == 'dev' and status == 'eval')):
                            output = model((image))[-1]
                            
                            if (isinstance(output, tuple) or isinstance(output, list)):
                                output = output[-1]
                                                         
                            loss = criterion(output, label)  
                            preds_th =output

                            if phase == 'train' or (phase == 'dev' and status == 'eval'):
                                if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)
                                loss.backward()  
                                optimizer.step()
                            
                            list_preds.append(preds_th.cpu())
                            list_label.append(label.cpu()) 

                        running_loss += loss.item() * image.size(0)
                        batch_pred_th = preds_th.data.cpu()
                        batch_true = label.data.cpu().numpy()
                        batch_pred_th = batch_pred_th.round()
                        batch_acc = accuracy_score(batch_true, batch_pred_th.numpy()) * 100                  
                        postfix_str = 'batch_loss: {:.03f}, batch_acc: {:.03f}'.format(loss.item(), batch_acc)
                        t.set_postfix_str(postfix_str)
                        t.update()
                            
                epoch_loss = running_loss / dataset_sizes[phase]
                
                y_pred = torch.cat(list_preds, dim=0)
                y_true = torch.cat(list_label, dim=0)
                
                y_score = y_pred.clone()
                y_pred = y_pred.round()
                epoch_acc = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy()) * 100  
                top5 = top_k_accuracy_score(y_true.detach().numpy(), y_score.detach().numpy(),k=5) * 100
                              
                logger.info('{} Loss: {:.4f}, Acc: {:.4f}, Top-5-Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, top5))
                
                if(phase == 'train'):
                    top1_train.append(epoch_acc)
                    top5_train.append(top5)
                elif(phase == 'test'):
                    top1_test.append(epoch_acc)
                    top5_test.append(top5)
                
                if parallel:
                    num_params = 0
                    for reshape_layer in model.module.reshape_layers:
                        num_params += count_parameters(reshape_layer)

                    num_params += count_parameters(model.module.fusion_net)
                    logger.info("Fusion Model Params: {}".format(num_params) )

                    genotype = model.module.genotype()
                else:
                    num_params = 0
                    genotype = 0

                logger.info(str(genotype))

                if phase == 'train' and epoch_loss != epoch_loss:
                    logger.info("Nan loss during training, escaping")
                    model.eval()              
                    return best_acc
                
                if phase == 'dev':
                    if epoch_acc > best_acc:
                        print("Updating the best_dev_acc")
                        best_epoch = epoch
                        best_acc = epoch_acc

                if phase == 'test':
                    if epoch_acc > best_test_acc:
                        print("Updating the best_test_acc")
                        best_test_acc = epoch_acc
                        best_test_genotype = copy.deepcopy(genotype)
                        best_test_epoch = epoch
                    
                        if parallel:
                            save(model.module, os.path.join(args.save, args.name+'.pt'))
                        else:
                            save(model, os.path.join(args.save, args.name+'.pt'))
                        
                        print("Best saved")

            file_name = "epoch_{}".format(epoch)
            file_name = os.path.join(args.save, "architectures", file_name)

            logger.info("Current best dev Acc: {}, at training epoch: {}".format(best_acc, best_epoch) )
            logger.info("Current best test Acc: {}, at training epoch: {}".format(best_test_acc, best_test_epoch) )

        if best_acc != best_acc and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            logger.info('Recording a NaN Acc, training for one more epoch.')
        else:
            failsafe = False
            
        cont_overloop += 1
    
    if best_acc != best_acc:
        best_acc = 0.0

    if status == 'search':
        return best_acc, best_genotype
    else:
        return best_test_acc,  top1_train, top5_train, top1_test, top5_test


def plot_and_save_confusion_matrix(cm, labels, filename):
    plt.figure(figsize=(6, 5))  # Smaller figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 16, "weight": "bold"}, linewidths=.5, square=True)
    plt.xlabel('Predicted Class', fontsize=14, weight='bold')
    plt.ylabel('True Class', fontsize=14, weight='bold')
    plt.xticks(np.arange(len(labels)) + 0.5, labels, fontsize=12, weight='bold', rotation=0)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, fontsize=12, weight='bold', rotation=0)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def test_uni_track_acc(model, criterion, dataloaders, device, phase='test', status='eval', modal_num=0, warmup_iters=5, cm_filename='confusion_matrix.png'):
    top1_test = []
    top5_test = []
    model.eval()  # Set model to evaluate mode
    list_preds = []
    list_label = []
    prediction_times = []
    sample_count = len(dataloaders[phase].dataset)  # Get the number of samples in the dataset
    print(f"Number of samples in the {phase} dataset: {sample_count}")
    with tqdm(dataloaders[phase], desc=f"{phase} phase") as t:
        for i, data in enumerate(dataloaders[phase]):
            image, label = data[modal_num], data[-1]
            image = image.to(device).float()
            label = label.to(device).float()

            start_time = time.time()
            with torch.no_grad():
                output = model(image)[-1]
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                loss = criterion(output, label)
            end_time = time.time()

            # Skip warm-up iterations for timing
            if i >= warmup_iters:
                prediction_times.append(end_time - start_time)

            preds_th = output
            list_preds.append(preds_th.cpu())
            list_label.append(label.cpu())

            batch_pred_th = preds_th.data.cpu()
            batch_true = label.data.cpu().numpy()
            batch_pred_th = batch_pred_th.round()
            batch_acc = accuracy_score(batch_true, batch_pred_th.numpy()) * 100
            postfix_str = f'batch_loss: {loss.item():.03f}, batch_acc: {batch_acc:.03f}'
            t.set_postfix_str(postfix_str)
            t.update()

    y_pred = torch.cat(list_preds, dim=0)
    y_true = torch.cat(list_label, dim=0)

    y_score = y_pred.clone()
    y_pred = y_pred.round()
    epoch_acc = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy()) * 100
    top5 = top_k_accuracy_score(y_true.detach().numpy(), y_score.detach().numpy(), k=5) * 100

    top1_test = epoch_acc
    top5_test = top5

    # Count the number of labels that are 0 and 1
    count_0 = torch.sum(y_true == 0).item()
    count_1 = torch.sum(y_true == 1).item()
    print(f"Number of true labels that are 0: {count_0}")
    print(f"Number of true labels that are 1: {count_1}")

    # Plot and save the confusion matrix
    cm = confusion_matrix(y_true.detach().numpy(), y_pred.detach().numpy())
    print(f"Confusion Matrix:\n{cm}")
    labels = ['Drone', 'Non-Drone']  # Adjust according to your dataset's labels
    plot_and_save_confusion_matrix(cm, labels, cm_filename)

    precision_ = precision_score(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=0)
    average_precision_ = average_precision_score(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=0)
    recall_ = recall_score(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=0)
    f1_score_ = f1_score(y_true.detach().numpy(), y_pred.detach().numpy(), pos_label=0)
    f1_macro_ = f1_score(y_true.detach().numpy(), y_pred.detach().numpy(), average='macro')

    avg_prediction_time = sum(prediction_times) / len(prediction_times) if prediction_times else float('nan')

    results = {
        'top-1_acc': top1_test,
        'top-5_acc': top5_test,
        'precision': precision_, 
        'average_precision': average_precision_,
        'recall': recall_,
        'f1_score': f1_score_,
        'f1_macro': f1_macro_,
        'avg_prediction_time': avg_prediction_time
    }

    return results

class Architect(object):
    def __init__(self, model, args, criterion, optimizer):
        self.network_weight_decay = args.weight_decay
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
    
    def log_learning_rate(self, logger):
        for param_group in self.optimizer.param_groups:
            logger.info("Architecture Learning Rate: {}".format(param_group['lr']))
            break
    
    def step(self, input_valid, target_valid, logger):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()