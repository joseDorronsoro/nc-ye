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
LR_SGD = 0.0184     #as used in Papyan; about 1.e-2
LR_DECAY = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
NUM_CLASSES = 10
MINIMUM_LR_FACTOR = 1.e-2
LR_ADAMW = 1.e-5 # better use Papyan's and multiply it but an appropriate lr_factor, e.g. 1.e-3
#MOMENTUM_ADAMW = 0.9 #use defaults
WEIGHT_DECAY_ADAMW = 1.e-1 #5.e-4 


def get_learning_rate(loss_name, lrate_factor):
    """
    Return the base learning rate associated with the selected loss.

    Learning rates follow the values used in the original
    Neural Collapse experiments and can be scaled when needed.
    """
    if loss_name == 'CrossEntropyLoss':
        return 0.0679

    if lrate_factor == 999.:
        return LR_SGD / 2.8461

    return LR_SGD * lrate_factor


#________ resnet18 model with frozen lhl weights, bias ________________________
#________ new functions: build_model_frozen_lhl, build_optimizer_frozen_lhl
def build_model(num_classes=NUM_CLASSES,
                input_channels=1,
                lhl_weights=None,
                lhl_bias=None,
                frozen_weights=False,
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
    #print('fc shape', model.fc.weight.shape, model.fc.bias.shape)
    
    # Freeze ONLY the final classification layer (weights and bias)
    if frozen_weights==True:
        if lhl_weights is None or lhl_bias is None:
            raise TypeError("weights and bias must be numpy arrays or torch tensors.")
        
        lhl_weights=torch.from_numpy(lhl_weights)
        lhl_bias=torch.from_numpy(lhl_bias)
        
        with torch.no_grad():
            model.fc.weight.copy_(lhl_weights)
            model.fc.bias.copy_(lhl_bias)

        # although trainable, fc weights/bias will have a small learning rate
        # not efficient but keep it for the time bwing
        for param in model.parameters():
            param.requires_grad = True
    
    #model.fc.weight.requires_grad = False
    #model.fc.bias.requires_grad = False

    # Papyan adjustments 
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



def build_optimizer(
        model,
        loss_name='MSELoss',
        optimizer_name='sgd',
        lr_factor=1.,
        weight_decay=WEIGHT_DECAY,
        epochs=350,
        frozen_weights=False):
    """
    Construct the optimizer and learning-rate scheduler.
    """
    lr = get_learning_rate(loss_name,
                           lr_factor)
                           
    if frozen_weights == True:
        fc_params = set(map(id, model.fc.parameters()))
    
        other_params = [
            p for p in model.parameters()
            if id(p) not in fc_params
        ]
    
        #first try; lr = 0. for fc params
        params = [{'params': other_params, 'lr': lr}, 
             {'params': model.fc.parameters(), 'lr': 1.e-3 * lr},
             ]
    
    else:
        #Sseems to work fine
        params = [{'params': model.parameters(), 'lr': lr}, 
             ]
        
    if optimizer_name == 'sgd':

        optimizer = optim.SGD(
            params=params, 
            momentum=MOMENTUM,
            weight_decay=weight_decay,
        )

        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer,
        #    milestones=LR_MILESTONES,
        #    gamma=LR_DECAY
        #)
        
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min= lr * MINIMUM_LR_FACTOR
        )

    # no optimizer other than SGD has been tested
    elif optimizer_name == 'adam':

        optimizer = optim.Adam(
            model.parameters(),
            weight_decay=WEIGHT_DECAY
        )

        scheduler = None

    elif optimizer_name == 'adamw':

        optimizer = optim.AdamW(
            # Instantiate AdamW without filtering out frozen parameters for now
            params = [{'params': model.parameters(), 'lr': lr_factor * LR_ADAMW}], 
            #momentum=MOMENTUM_ADAMW,
            weight_decay=weight_decay, #WEIGHT_DECAY_ADAMW
        )

        #scheduler = None
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min= lr * MINIMUM_LR_FACTOR
        )


    elif optimizer_name == 'rmsp':

        optimizer = optim.RMSprop(
            model.parameters(),
            weight_decay=WEIGHT_DECAY
        )

        scheduler = None

    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    return optimizer, scheduler
    
    
#________ adapting:
#         train_fn: new argument frozen_lhl as a flag
#         train_loop: new argument frozen_lhl as a flag,
#               calls adapteed train_fn;
#               if frozen_lhl False, training should revert to that of plain launcher.py
def train_fn(model, criterion, device, num_classes, train_loader, optimizer, batch_size, 
        #frozen_lhl=False
    ):
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
        # not clear why do I want this
        #if len(target.shape) == 1:
        #    accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
        #elif len(target.shape) == 2 and target.shape[1] == num_classes:
        #    target_class = torch.argmax(target,dim=1)
        #    accuracy = torch.mean((torch.argmax(out,dim=1)==target_class).float()).item()
        
    #print('checking grad norms')
    ## Check if gradients are reaching the layer right before fc
    ## does one print per epoch; remove later
    #print("fc.weight norm; should stay constant:", model.fc.weight.norm())
    #print("Layer4 grad norm; should eventually decrease:", model.layer4[1].conv2.weight.grad.abs().mean().item())
        
    return 



def train_loop(model,
               train_loader,
               #train_loader_resampled,
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
               #frozen_lhl=False
               ):
    """
    Execute the complete training procedure.
    Uses train_loader_resampled, which may have less than full samples
    per class, and plain train_loader to evaluate.

    Tracks training MSE and periodically reports majority and
    minority class accuracies.
    """
    mse_history = []

    for epoch in range(0, epochs + 1):

        train_fn(
            model,
            criterion,
            device,
            NUM_CLASSES,
            train_loader,
            optimizer,
            #epoch,
            train_loader.batch_size,
            #frozen_lhl=frozen_lhl
        )

        if scheduler is not None:
            scheduler.step()

        #also uses resampling loader: actual mse    
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

        if epochs <= 10 or epoch % 50 == 0:
            print(
                f'\nepoch={epoch:4d} '
                f'mse={mse:.6f}'
            )

            #uses plain loader    
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
            
            print('..... checking grad norms')
            print("\tfc.weight norm; should stay near constant when freezing weights:", model.fc.weight.norm().item())
            print("\tLayer4 abs grad mean; should eventually decrease:", model.layer4[1].conv2.weight.grad.abs().mean().item())
    
            
    print("\n\nFinal scores")
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
    
#______________________________________________________________________________ end frozen


def build_criterion(loss_name):
    """
    Create the training loss function.
    """
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()

    return nn.MSELoss()



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


# ------------------------------------------------------------------------------ old stuff, remove eventually
#def build_model_00(num_classes=NUM_CLASSES,
#                input_channels=1,
#                lhl_weights=None,
#                lhl_bias=None,
#                device='cpu'):
#    """
#    Build the modified ResNet18 backbone used in the experiments.
#
#    The model uses a reduced first convolution and removes the
#    initial max-pooling layer to better suit MNIST-sized images.
#    """
#    model = models.resnet18(
#        weights=None,
#        num_classes=num_classes
#    )
#
#    model.conv1 = nn.Conv2d(
#        input_channels,
#        model.conv1.out_channels,
#        kernel_size=3,
#        stride=1,
#        padding=1,
#        bias=False
#    )
#
#    model.maxpool = nn.Identity()
#
#    return model.to(device)
#
#
#
#def build_optimizer_00(
#        model,
#        loss_name='MSELoss',
#        optimizer_name='sgd',
#        lr_factor=1.):
#    """
#    Construct the optimizer and learning-rate scheduler.
#    """
#    lr = get_learning_rate(loss_name,
#                      lr_factor)
#
#    if optimizer_name == 'sgd':
#
#        optimizer = optim.SGD(
#            model.parameters(),
#            lr=lr,
#            momentum=MOMENTUM,
#            weight_decay=WEIGHT_DECAY
#        )
#
#        scheduler = optim.lr_scheduler.MultiStepLR(
#            optimizer,
#            milestones=LR_MILESTONES,
#            gamma=LR_DECAY
#        )
#
#    elif optimizer_name == 'adam':
#
#        optimizer = optim.Adam(
#            model.parameters(),
#            weight_decay=WEIGHT_DECAY
#        )
#
#        scheduler = None
#
#    elif optimizer_name == 'adamw':
#
#        optimizer = optim.AdamW(
#            model.parameters(),
#            weight_decay=WEIGHT_DECAY
#        )
#
#        scheduler = None
#
#    elif optimizer_name == 'rmsp':
#
#        optimizer = optim.RMSprop(
#            model.parameters(),
#            weight_decay=WEIGHT_DECAY
#        )
#
#        scheduler = None
#
#    else:
#        raise ValueError(f'Unknown optimizer: {optimizer_name}')
#
#    return optimizer, scheduler


