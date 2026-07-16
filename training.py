import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

from sklearn.metrics import mean_squared_error

from analysis import (
    #majority_minority_accuracy,
    evaluate_model,
)

LR_MILESTONES = [150, 300, 600, 1200, 2400]
LR_DECAY = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 10


def build_model(num_classes=NUM_CLASSES,
                input_channels=1,
                device='cpu'):
    """
    Build the modified ResNet18 backbone used in the experiments.

    The model uses a reduced first convolution and removes the
    initial max-pooling layer to better suit MNIST-sized images.
    """
    model = models.resnet18(
        weights=None,
        num_classes=num_classes
    )

    model.conv1 = nn.Conv2d(
        input_channels,
        model.conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    return model.to(device)



def get_learning_rate(loss_name, lrate_factor):
    """
    Return the base learning rate associated with the selected loss.

    Learning rates follow the values used in the original
    Neural Collapse experiments and can be scaled when needed.
    """
    if loss_name == 'CrossEntropyLoss':
        return 0.0679

    if lrate_factor == 999.:
        return 0.0184 / 2.8461

    return 0.0184 * lrate_factor



def build_optimizer(
        model,
        loss_name='MSELoss',
        optimizer_name='sgd',
        lr_factor=1.):
    """
    Construct the optimizer and learning-rate scheduler.
    """
    lr = get_learning_rate(loss_name,
                      lr_factor)

    if optimizer_name == 'sgd':

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=LR_MILESTONES,
            gamma=LR_DECAY
        )

    elif optimizer_name == 'adam':

        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=WEIGHT_DECAY
        )

        scheduler = None

    elif optimizer_name == 'adamw':

        optimizer = optim.AdamW(
            model.parameters(),
            weight_decay=WEIGHT_DECAY
        )

        scheduler = None

    elif optimizer_name == 'rmsp':

        optimizer = optim.RMSprop(
            model.parameters(),
            weight_decay=WEIGHT_DECAY
        )

        scheduler = None

    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    return optimizer, scheduler



def build_criterion(loss_name):
    """
    Create the training loss function.
    """
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()

    return nn.MSELoss()



def train_fn(model, criterion, device, num_classes, train_loader, optimizer, epoch, batch_size):
    """
    Train the model for one epoch using a single pass through
    the training DataLoader.
    """    
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
            if batch_idx == 0: 
                print('incorrect batch_sizes:', batch_size, data.shape[0])
                sys.exit(1)
            else:
                #probably the last batch is smaller than batch_size
                continue
        
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
            if len(target.shape) == 1:
                loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())
            elif len(target.shape) == 2 and target.shape[1] == num_classes:
                loss = criterion(out, target.float())
            else:
                sys.exit('something wrong in train ...')
        
        loss.backward()
        optimizer.step()

        #accs are computed at each step; global value should be obtained elsewhere after training
        #this acc should work in all cases ...
        if len(target.shape) == 1:
            accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
        elif len(target.shape) == 2 and target.shape[1] == num_classes:
            target_class = torch.argmax(target,dim=1)
            accuracy = torch.mean((torch.argmax(out,dim=1)==target_class).float()).item()
            
    return 



def bbn_predict(model, loader, device):
    """
    Compute backbone predictions for all samples in a DataLoader.

    Returns model outputs together with the corresponding targets.
    """
    model.to(device)
    model.eval()
    
    l_targ_out = []
    l_out = []
    
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        
        l_targ_out.append(target.cpu().detach().numpy())
        output = model(data)
        l_out.append(output.cpu().detach().numpy())
    
    
    targ_out  = np.concatenate(l_targ_out, axis=0)
    model_out = np.concatenate(l_out, axis=0)
    
    return model_out, targ_out



def train_loop(model,
               train_loader,
               test_loader,
               criterion,
               optimizer,
               scheduler,
               epochs,
               encoding,
               device,
               train_fn,
               model_targ_out,
               bbn_predict,
               ye_classifier,
               ):
    """
    Execute the complete training procedure.

    Tracks training MSE and periodically reports majority and
    minority class accuracies.
    """
    mse_history = []

    for epoch in range(1, epochs + 1):

        train_fn(
            model,
            criterion,
            device,
            NUM_CLASSES,
            train_loader,
            optimizer,
            epoch,
            train_loader.batch_size
        )

        if scheduler is not None:
            scheduler.step()

        targets, outputs = model_targ_out(
            model,
            train_loader,
            device
        )

        mse = mean_squared_error(
            targets,
            outputs
        )

        mse_history.append(mse)

        if epoch == 1 or epoch % 50 == 0:
            #print("Calling evaluate_epoch()")
            maj_acc, min_acc = evaluate_model(
                model,
                train_loader,
                test_loader,
                encoding,
                device,
                bbn_predict,
            )

            print(
                f'epoch={epoch:4d} '
                f'mse={mse:.6f}'
            )

            print(f'..... test majority acc. {maj_acc:.4f}')
            print(f'..... test minority acc. {min_acc:.4f}',
                  flush=True)
    
    print("Final scores")
    print(
        f'mse={mse:.6f}'
    )

    maj_acc, min_acc = evaluate_model(
                model,
                train_loader,
                test_loader,
                encoding,
                device,
                bbn_predict,
            )

    print(f'..... test majority acc. {maj_acc:.4f}')
    print(f'..... test minority acc. {min_acc:.4f}',
          flush=True)

    return mse_history


def model_targ_out(model, loader, device):
    """
    Return targets and model outputs for all samples in a DataLoader.
    """
    model.eval()
    
    l_targs = []
    l_out = []
    
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        
        l_targs.append(target.cpu().detach().numpy())
    
        output = model(data)
        l_out.append(output.cpu().detach().numpy())
        
    targs_out = np.concatenate(l_targs, axis=0)
    model_out = np.concatenate(l_out, axis=0)
    
    return targs_out, model_out


def get_activation(name, activation):
    """
    Create a forward hook used to capture intermediate activations.
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



def model_lhl_out_data(model, loader, device, layer='avgpool'):
    """
    Extract last-hidden-layer representations and classifier parameters.

    Returns targets, model outputs, hidden-layer activations and the
    weights and biases of the final classifier layer.
    """
    model.eval()
    activation = {}
    layer_names = [n for n, _ in model.named_children()]
    
    for ly in layer_names:
        #print("model." + ly + ".register_forward_hook(get_activation('" + ly + "'))")
        exec("model." + ly + ".register_forward_hook(get_activation('" + ly + "', activation))")
        
    #print(layer_names, activation.keys())
    
    l_targs = []
    l_out = []
    l_h = []
    
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        
        l_targs.append(target.cpu().detach().numpy())
    
        output = model(data)
        l_out.append(output.cpu().detach().numpy())
        
        h = activation[layer]
        l_h.append(h.cpu().detach().numpy())
    
    targs_out = np.concatenate(l_targs, axis=0)
    model_out = np.concatenate(l_out, axis=0)
    layer_out = np.concatenate(l_h, axis=0)
    
    w_lhl = model.fc.weight.cpu().detach().numpy()
    b_lhl = model.fc.bias.cpu().detach().numpy()
    
    return targs_out, model_out, \
           layer_out.reshape((layer_out.shape[0], layer_out.shape[1])), \
           (w_lhl, b_lhl)
