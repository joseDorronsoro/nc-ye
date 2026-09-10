import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

from sklearn.metrics import mean_squared_error

import fisher_functions as ff

from analysis import (
    evaluate_model,
)

NUM_CLASSES = 10

LR_SGD = 0.0184     #as used in Papyan; about 1.e-2
WEIGHT_DECAY = 5.e-4
HREG_DECAY = 5.e-4
MOMENTUM = 0.9

#LR_ADAMW use Papyan's and multiply it but an appropriate lr_factor, e.g. 1.e-3
WEIGHT_DECAY_ADAMW = 1.e-1 #5.e-4 
#MOMENTUM_ADAMW: use defaults

#lr_scheduler
LR_MILESTONES = [150, 300, 600, 1200, 2400]
LR_DECAY = 0.1

#cosine_scheduler
MINIMUM_LR_FACTOR = 1.e-2

# ------- frozen weight matrix ------------------------------------------------
def probs(frac):
    """Imbalanced probabilities
    """
    pr = np.array(5 * [1.] + 5 * [float(frac)])
    pr = pr / pr.sum()
    
    return pr 



def householder_V(pr):
    """Householder V matrix for the SVD of h_pi
    """
    v = (np.sqrt(pr) + np.array(9 * [0.] + [1.])).reshape(-1, 1)
    V = (2. / (v.T @ v)) * (v @ v.T) - np.eye(len(pr))
    
    return V



def h_pi(pr):
    """H_pi matrix from probs.
    
    Good for checking that froW bwloNot used.
    """
    H_pi = np.eye(10) - np.sqrt(pr).reshape(-1, 1) @ np.sqrt(pr ).reshape(-1, 1).T

    return H_pi



def optimal_W(dim, frac, verbose=False):
    """Frozen weight matrix for Y targets
    """
    pr = probs(frac)
    V = householder_V(pr)
    
    J = np.eye(len(pr))
    J[-1, -1] = 0.
    
    U = np.eye(dim)[ : , : len(pr)]
    
    if verbose:
        print('\n' + 10 * '.' + ' checking optimal weight matrix')
        print('U.T @ U:', np.allclose(U.T @ U, np.eye(len(pr))))
        print('V @ V.T:', np.allclose(V @ V.T, np.eye(len(pr))))
        print('diag J', np.diagonal(J))
        print('h_pi svd', np.allclose(V @ J @ V.T, h_pi(pr)), '\n')
        #print(np.linalg.norm(U @ J @ V.T))
    
    return U @ J @ V.T


# subspace projection function
def  project_Q_2_W(Q, W):
    Q_proj = W @ np.linalg.pinv(W, rcond=1.e-3) @ Q
    return Q_proj
    

# -----------------------------------------------------------------------------


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


def build_model(resnet_model='18',
                num_classes=NUM_CLASSES,
                input_channels=1,
                lhl_weights=None,
                lhl_bias=None,
                #lhl_dim=512,
                enlarge_lhl=False,
                device='cpu'):
    """
    Build the modified ResNet18 backbone used in the experiments.

    The model uses a reduced first convolution and removes the
    initial max-pooling layer to better suit MNIST-sized images.
    
    EXPERIMENTAL:
        1. initialize weights as optimal and train
    """
    if resnet_model == '18':
        model = models.resnet18(
            weights=None,
            num_classes=num_classes
        )
    elif resnet_model == '34':
        model = models.resnet34(
            weights=None,
            num_classes=num_classes
        )
    elif resnet_model == '50':
        model = models.resnet50(
            weights=None,
            num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Invalid model '{resnet_model}'. Expected one of 18, 34 or 50."
        )
    
    print(5*'.' + f" Using resnet model {resnet_model}.")
    lhl_dim = model.fc.in_features
    
    if lhl_bias is not None and lhl_weights is not None:
        #print('initializing weights................................................................')
        lhl_weights=torch.from_numpy(lhl_weights)
        lhl_bias=torch.from_numpy(lhl_bias)
        
        with torch.no_grad():
            model.fc.weight.copy_(lhl_weights)
            model.fc.bias.copy_(lhl_bias)
    
    # fc weights/bias fully trainable; later, try freezing lhl weights
    # adapt effect with possibly adjusted learning rates
    for param in model.parameters():
        param.requires_grad = True
       
    #for later: do not compute weigh and bias grads
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
    
    # Replace model.fc: perhaps artificial, as the "true" lhl dime is 512 
    if enlarge_lhl == True:
        model.fc = nn.Sequential(
            nn.Linear(512, lhl_dim),
            nn.BatchNorm1d(lhl_dim),
            nn.ReLU(),
            nn.Linear(lhl_dim, num_classes)
        )

    return model.to(device)



def build_optimizer(
        model,
        loss_name='MSELoss',
        optimizer_name='sgd',
        lr_factor=1.,
        weight_decay=WEIGHT_DECAY,
        epochs=350,
        warmup_epochs=0
        ):
    """
    Construct the optimizer and learning-rate scheduler.
    """
    lr = get_learning_rate(loss_name,
                           lr_factor)
    print('effective learning-rate', lr)
    
    # grad for all params
    params = [{'params': model.parameters(), 'lr': lr}, 
             ]
        
    if optimizer_name == 'sgd':

        optimizer = optim.SGD(
            params=params, 
            momentum=MOMENTUM,
            weight_decay=weight_decay,
        )

    elif optimizer_name == 'adamw':

        optimizer = optim.AdamW(
            # Instantiate AdamW without filtering out frozen parameters for now
            params = params, #[{'params': model.parameters(), 'lr': lr}], 
            weight_decay=weight_decay, 
        )

    # no other optimizer has been tested
    elif optimizer_name == 'adam':

        optimizer = optim.Adam(
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

    
    warmup_epochs = min(warmup_epochs, epochs // 3)
    print('effective warmup_epochs', warmup_epochs)

    if warmup_epochs > 0:
        print(' using warmup scheduling with ', warmup_epochs, 
            'epochs plus cosine scheduler with ', epochs - warmup_epochs, 'epochs', 
            flush=True
        )
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 0.1,  
            end_factor = 1.0,     
            total_iters = warmup_epochs
        )
        
        # 3. Cosine Scheduler: Decays LR from peak_lr down to eta_min over remaining 45 epochs
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = epochs - warmup_epochs,  
            eta_min = lr * MINIMUM_LR_FACTOR
        )
    
        # 4. Combine into a single SequentialLR
    
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]  
        )
    
    else:
        print(5*'.' + ' using cosine scheduling with no warmup')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = epochs,  
            eta_min = lr * MINIMUM_LR_FACTOR
        )
    
    return optimizer, scheduler
    
    
def train_fn(model, criterion, device, num_classes, train_loader, optimizer, batch_size, 
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
        
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        
        out = model(data)
        
        if str(criterion) == 'CrossEntropyLoss()':
            loss = criterion(out, target)
        
        elif str(criterion) == 'MSELoss()' or 'AnchorLoss' in str(criterion):
            if len(target.shape) == 1:
                loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())
            elif len(target.shape) == 2 and target.shape[1] == num_classes:
                loss = criterion(out, target.float())
            else:
                sys.exit('something wrong in train ...')
        
        elif 'CenterLoss' in str(criterion):
            loss = criterion(data, target)
        
        elif 'HRegLoss' in str(criterion):
            loss = criterion(data, target)
        
        loss.backward()
        
        optimizer.step()

    return 



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
               cfg
               ):
    """
    Execute the complete training procedure.
    Uses train_loader_resampled, which may have less than full samples
    per class, and plain train_loader to evaluate.

    Tracks training MSE and periodically reports majority and
    minority class accuracies.
    """
    mse_history = []

    #print("Is memory address identical?", id(criterion.layer.weight) == id(model.fc.weight))

    # Will raise an AssertionError if they point to different memory addresses
    if 'AnchorLoss' in str(criterion):
        assert id(criterion.layer.weight) == id(model.fc.weight), \
            "CRITICAL: criterion.layer is pointing to an orphaned weight matrix!"
    
    pr = probs(cfg.frac)
    H_pi = h_pi(pr)  
    
    dim = model.fc.weight.shape[1]
    W_target = torch.Tensor(optimal_W(dim, cfg.frac, verbose=False).T)
    
    for epoch in range(0, epochs + 1):
        train_fn(
            model,
            criterion,
            device,
            NUM_CLASSES,
            train_loader,
            optimizer,
            train_loader.batch_size,
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

        if epochs <= 10 or epoch % 50 == 0:
            print(
                f'\nepoch={epoch:4d} '
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
            print(f'..... test minority acc. {min_acc:.4f}')
            
            #print('..... checking grad norms')
            #print("\tfc.weight norm; ideally, final fc weights should have a norm near 3.0, that of the optimal W solution:", model.fc.weight.norm().item())
            #print("\tLayer4 abs grad mean; should eventually decrease:", model.layer4[1].conv2.weight.grad.abs().mean().item(), flush=True)
                  
            with torch.no_grad():
                # Calculate distance from target matrix
                weight_drift = torch.norm(model.fc.weight - W_target.to('cuda'), p='fro').item()
                b_norm = torch.norm(model.fc.bias, p='fro').item()
                print(f"........... ||W_fc - W||: {weight_drift:.4f} \t ||b||: {b_norm:.4f}") 
                
                train_results = model_lhl_out_data(model, train_loader, device, layer='avgpool')
                (
                    targs_out,
                    model_out,
                    lhl_out,
                    w_b_lhl,
                ) = train_results

                #class centers diff vs h_pi
                label = np.argmax(targs_out, axis=1)
                ccm  = ff.ccm_array(lhl_out,  label)
                q = ccm @ np.diag(np.sqrt(pr)) 
                qdn = np.linalg.norm(q.T @ q - H_pi)
                print(f'..... ||q.T @ q - H_pi||: {qdn:.4f}', flush=True)
    
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


#______________________________________________________________________________

class HRegLoss(nn.Module):
    """
    Center Loss criterion matching standard PyTorch loss signature (inputs, targets).
    
    Automatically hooks into the model's final linear layer to capture 
    last hidden layer features (h) during model(inputs).
    """
    def __init__(self, model: nn.Module, 
                 fc_layer_name: str = 'fc', 
                 hreg_decay: float = 5.e-4,
                 reduction: str = 'mean'):
        super().__init__()
        self.model = model
        self.layer = getattr(model, fc_layer_name)
        
        self.hreg_decay = hreg_decay
        if self.hreg_decay > 0.:
            print('\n' + 5 * '.' + ' using HRegLoss')
        else:
            print('\n' + 5 * '.' + ' using plain MSE')
        
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        self.current_h = None
        
        # #Locate classification layer (e.g., model.fc)
        self.fc_layer = getattr(model, fc_layer_name)
        
        # Automatically attach hook to capture 'h'
        self.layer.register_forward_hook(self._hook_fn)
        

    def _hook_fn(self, module, input, output):
        # Intercept input tensor entering fc layer
        self.current_h = input[0]

    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input batch 'x' (matches standard criterion interface)
            targets (torch.Tensor): Class labels 'y' [batch_size]
        """
        outputs = self.model(inputs).to(dtype=torch.float)
        targets = targets.to(dtype=torch.float)
        mse = self.mse_loss_fn(outputs, targets)#.to(dtype=torch.float)
 
        if self.hreg_decay > 0.:
            if self.current_h is None:
                raise RuntimeError("You must execute `model(inputs)` before calling CenterLoss or AnchorCenterLoss.forward().")
                
            h = self.current_h.to(dtype=torch.float)
            
            # Average squared L2 norm per sample in the batch
            #hreg_loss = torch.mean(torch.sum(h ** 2, dim=1))
            hreg_loss = torch.sum(h ** 2) / h.shape[0]

            # Clear cached features after computing loss
            self.current_h = None
            
            return mse + self.hreg_decay * hreg_loss 
        
        else:
            return mse
        

#______________________________________________________________________________
class AnchorLoss(nn.Module):
    """
    Combined loss function that calculates MSE loss between predictions and targets,
    plus an L2 distance penalty anchoring a weight tensor to a fixed target matrix W.
    """
    def __init__(self, 
                 model: nn.Module,
                 fc_layer_name: str = 'fc', 
                 frac: float = 0.005, 
                 dim: int = 512, 
                 lambda_anchor: float = 0.01, 
                 reduction: str = 'mean'):
        super().__init__()
        
        self.model = model
        self.layer = getattr(model, fc_layer_name)
        
        self.frac = frac 
        self.lambda_anchor = lambda_anchor
        if self.lambda_anchor > 0.:
            print('\n' + 5 * '.' + ' using AnchorLoss')
        else: 
            print('\n' + 5 * '.' + ' using plain MSE')
            
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        
        W_target = torch.Tensor(optimal_W(dim, frac, verbose=False)).detach()
        self.W_target = W_target.T
        
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
            current_weight (torch.Tensor): The active layer weight matrix (e.g., model.fc.weight).

        Returns:
            torch.Tensor: Combined scalar loss (MSE + Anchor Penalty).
        """
        # 1. Primary task loss
        mse = self.mse_loss_fn(outputs, targets)
 
        if self.lambda_anchor > 0.:
            # 2. Penalty distance from target matrix W
            anchor_penalty = torch.sum((self.layer.weight - self.W_target.to('cuda')) ** 2.) #+ torch.sum(self.layer.bias ** 2.) * self.layer.weight.shape[1] / 10
        
            # 3. Combined loss
            total_loss = mse + (self.lambda_anchor * anchor_penalty)
        
        else: 
            total_loss = mse
            
        return total_loss


#______________________________________________________________________________
class AnchorHRegLoss(nn.Module):
    """
    Combined Anchor and Center Loss
    Reduces to 
        AnchorLoss if hreg_decay <= 0. and lambda_anchor > 0.
        HRegLoss if hreg_decay > 0. and lambda_anchor <= 0.
        MSE if if hreg_decay <= 0. and lambda_anchor <= 0.
    
    Automatically hooks into the model's final linear layer to capture 
    last hidden layer features (h) during model(inputs).
    """
    def __init__(self, model: nn.Module, 
                 fc_layer_name: str = 'fc', 
                 frac: float = 0.005, 
                 dim: int = 512, 
                 hreg_decay: float = 5.e-4,
                 lambda_anchor: float = 0.01,
                 reduction: str = 'mean'):
        super().__init__()
        self.model = model
        
        self.frac = frac 
        
        # W_target
        W_target = torch.Tensor(optimal_W(dim, frac, verbose=False)).detach()
        self.W_target = W_target.to('cuda', dtype=torch.float).T
        
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        
        self.hreg_decay = hreg_decay
        
        self.current_h = None
        
        #Locate classification layer (e.g., model.fc)
        self.layer = getattr(model, fc_layer_name)
        
        # Automatically attach hook to capture 'h'
        self.layer.register_forward_hook(self._hook_fn)
        
        self.lambda_anchor = lambda_anchor
        
        if self.hreg_decay > 0. and self.lambda_anchor > 0.:
            #full AnchorHRegLoss
            print('\n' + 5 * '.' + ' using full AnchorHRegLoss')
        
        elif self.hreg_decay > 0.:
            #just HRegLoss
            print('\n' + 5 * '.' + ' using HRegLoss')
        
        elif self.lambda_anchor > 0.:
            #just AnchorLoss
            print('\n' + 5 * '.' + ' using AnchorLoss')
        
        else:
            #plain mse
            print('\n' + 5 * '.' + ' using plain MSE')
        

    def _hook_fn(self, module, input, output):
        # Intercept input tensor entering fc layer
        self.current_h = input[0]

    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input batch 'x' (matches standard criterion interface)
            targets (torch.Tensor): Class labels 'y' [batch_size]
        """
        outputs = self.model(inputs).to(dtype=torch.float)
        targets = targets.to(dtype=torch.float)
        
        mse = self.mse_loss_fn(outputs, targets)
 
        #hreg penalty
        if self.current_h is None:
            raise RuntimeError("You must execute `model(inputs)` before calling CenterLoss or AnchorCenterLoss.forward().")
            
        h = self.current_h.to(dtype=torch.float)
        
        #mean of the squared norms of the batch's elements
        hreg_loss = torch.mean(torch.sum(h ** 2, dim=1)).to(dtype=torch.float) / h.shape[0] 
        
        # Clear cached features after computing loss
        self.current_h = None
        
        #anchor penalty
        anchor_penalty = torch.sum((self.layer.weight - self.W_target) ** 2.)
        
        if self.hreg_decay > 0. and self.lambda_anchor > 0.:
            #full AnchorHRegLoss
            return mse + self.hreg_decay * hreg_loss + self.lambda_anchor * anchor_penalty
        
        elif self.hreg_decay > 0.:
            #just HRegLoss
            return mse + self.hreg_decay * hreg_loss
        
        elif self.lambda_anchor > 0.:
            #just AnchorLoss
            return mse + self.lambda_anchor * anchor_penalty
        
        else:
            #plain mse
            return mse


#______________________________________________________________________________
class CenterLoss(nn.Module):
    """
    Center Loss criterion matching standard PyTorch loss signature (inputs, targets).
    
    Automatically hooks into the model's final linear layer to capture 
    last hidden layer features (h) during model(inputs).
    """
    def __init__(self, model: nn.Module, 
                 fc_layer_name: str = 'fc', 
                 frac: float = 0.005, 
                 dim: int = 512, 
                 lambda_center: float = 0.01,
                 reduction: str = 'mean'):
        super().__init__()
        self.model = model
        
        self.frac = frac 
        
        W_target = torch.Tensor(optimal_W(dim, frac, verbose=False)).detach()
        self.W_target = W_target.to('cuda', dtype=torch.float).T
        
        self.lambda_center = lambda_center
        
        if self.lambda_center > 0.:
            print('\n' + 5 * '.' + ' using full CenterLoss')
        else:
            print('\n' + 5 * '.' + ' using plain MSE')
        
        self.current_h = None
        
        # Locate classification layer (e.g., model.fc)
        self.layer = getattr(model, fc_layer_name)
        
        # Automatically attach hook to capture 'h'
        self.layer.register_forward_hook(self._hook_fn)
        
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        
        pr = probs(self.frac)
        self.centers = self.W_target.T @ torch.Tensor(np.diag(1. / np.sqrt(pr))).to('cuda', dtype=torch.float)
        
        if self.lambda_center > 0.:
            #full CenterLoss
            print('\n' + 5 * '.' + ' using CenterLoss')
        
        else:
            #plain mse
            print('\n' + 5 * '.' + ' using plain MSE')
                

    def _hook_fn(self, module, input, output):
        # Intercept input tensor entering fc layer
        self.current_h = input[0]


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input batch 'x' (matches standard criterion interface)
            targets (torch.Tensor): Class labels 'y' [batch_size]
        """
        outputs = self.model(inputs).to(dtype=torch.float)
        targets = targets.to(dtype=torch.float)
            
        mse = self.mse_loss_fn(outputs, targets).to(dtype=torch.float)
 
        if self.lambda_center > 0.:
            if self.current_h is None:
                raise RuntimeError("You must execute `model(inputs)` before calling CenterLoss.forward().")
                
            h = self.current_h.to(dtype=torch.float)
        
            local_labels = torch.Tensor(np.argmax(targets.cpu().numpy(), axis = 1)).to(dtype=torch.int)
            unique_classes, counts = torch.unique(local_labels, return_counts=True)
        
            total_loss = 0.0
            valid_classes = 0
            
            min_samples_threshold = 1  # Ignore centers calculated from < 3 samples
            
            for c, count in zip(unique_classes, counts):
                if count < min_samples_threshold:
                    continue  # Skip noisy minority estimates
                    
                mask = (local_labels == c)
                h_c = h[mask]                     # Features for class c
                W_c = self.centers.T[c]     # Target weight for class c
                
                # Center loss for this specific class
                class_loss = torch.mean(torch.sum((h_c - W_c) ** 2, dim=1))
                total_loss += class_loss
                valid_classes += 1
                
            if valid_classes == 0:
                center_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else:
                center_loss = (total_loss / valid_classes).to(dtype=h.dtype)
        
            # Clear cached features after computing loss
            self.current_h = None
        
        if self.lambda_center > 0.:
            #just CenterLoss
            return mse + self.lambda_center * center_loss
            
        else:
            #plain mse
            return mse

#______________________________________________________________________________

class AnchorCenterLoss(nn.Module):
    """
    Center Loss criterion matching standard PyTorch loss signature (inputs, targets).
    
    Automatically hooks into the model's final linear layer to capture 
    last hidden layer features (h) during model(inputs).
    """
    def __init__(self, model: nn.Module, 
                 fc_layer_name: str = 'fc', 
                 frac: float = 0.005, 
                 dim: int = 512, 
                 lambda_center: float = 0.01,
                 lambda_anchor: float = 0.01, 
                 reduction: str = 'mean'):
        super().__init__()
        self.model = model
        self.layer = getattr(model, fc_layer_name)
        
        self.frac = frac 
        
        W_target = torch.Tensor(optimal_W(dim, frac, verbose=False)).detach()
        self.W_target = W_target.to('cuda', dtype=torch.float).T
        self.dim = dim
        
        self.lambda_anchor = lambda_anchor
        self.lambda_center = lambda_center
        #print(20 * '.' + 'using lambda_anchor:', lambda_anchor)
        #print(20 * '.' + 'using lambda_center:', lambda_center)
        
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        pr = probs(self.frac)
        self.centers = self.W_target.T @ torch.Tensor(np.diag(1. / np.sqrt(pr))).to('cuda', dtype=torch.float)
        
        self.current_h = None
        
        # #Locate classification layer (e.g., model.fc)
        self.fc_layer = getattr(model, fc_layer_name)
        
        # Automatically attach hook to capture 'h'
        self.layer.register_forward_hook(self._hook_fn)
        
        if self.lambda_center > 0. and self.lambda_anchor > 0.:
            #full AnchorCenterLoss
            print('\n' + 5 * '.' + ' using full AnchorCenterLoss')
        
        elif self.lambda_center > 0.:
            #just CenterLoss
            print('\n' + 5 * '.' + ' using CenterLoss')
        
        elif self.lambda_anchor > 0.:
            #just AnchorLoss
            print('\n' + 5 * '.' + ' using AnchorLoss')
        
        else:
            #plain mse
            print('\n' + 5 * '.' + ' using plain MSE')
        
        
    def _hook_fn(self, module, input, output):
        # Intercept input tensor entering fc layer
        self.current_h = input[0]

    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input batch 'x' (matches standard criterion interface)
            targets (torch.Tensor): Class labels 'y' [batch_size]
        """
        outputs = self.model(inputs).to(dtype=torch.float)
        targets = targets.to(dtype=torch.float)
        mse = self.mse_loss_fn(outputs, targets)
 
        if self.lambda_anchor > 0.:
            #remove bias penalty??
            anchor_penalty = torch.sum((self.layer.weight - self.W_target.to('cuda')) ** 2.) + torch.sum(self.layer.bias ** 2.) * self.layer.weight.shape[1] / 10
        
        if self.lambda_center > 0.:
            if self.current_h is None:
                raise RuntimeError("You must execute `model(inputs)` before calling CenterLoss.forward().")
                
            h = self.current_h.to(dtype=torch.float)
        
            local_labels = torch.Tensor(np.argmax(targets.cpu().numpy(), axis = 1)).to(dtype=torch.int)
            unique_classes, counts = torch.unique(local_labels, return_counts=True)
        
            total_loss = 0.0
            valid_classes = 0
            
            min_samples_threshold = 1  # Ignore centers calculated from < 1 samples
            
            for c, count in zip(unique_classes, counts):
                if count < min_samples_threshold:
                    continue  # Skip noisy minority estimates
                    
                mask = (local_labels == c)
                h_c = h[mask]                     # Features for class c
                W_c = self.centers.T[c]     # Target weight for class c
                
                # Center loss for this specific class
                class_loss = torch.mean(torch.sum((h_c - W_c) ** 2, dim=1))
                total_loss += class_loss
                valid_classes += 1
                
            if valid_classes == 0:
                center_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else:
                center_loss = (total_loss / valid_classes).to(dtype=h.dtype)
        
            # Clear cached features after computing loss
            self.current_h = None
        
        if self.lambda_center > 0. and self.lambda_anchor > 0.:
            #full AnchorCenterLoss
            return mse + self.lambda_center * center_loss + self.lambda_anchor * anchor_penalty
        
        elif self.lambda_center > 0.:
            #just HRegLoss
            return mse + self.lambda_center * center_loss 
        
        elif self.lambda_anchor > 0.:
            #just AnchorLoss
            return mse + self.lambda_anchor * anchor_penalty
        
        else:
            #plain mse
            return mse

#______________________________________________________________________________

def build_criterion(loss_name, cfg, model):
    """
    Create the training loss function.
    """
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()

    elif loss_name == 'MSELoss':
        return nn.MSELoss(reduction='mean')
        
    elif loss_name == 'HRegLoss':
        return HRegLoss(model,
                fc_layer_name='fc', 
                hreg_decay=cfg.hreg_decay)

    elif loss_name == 'AnchorLoss':
        return AnchorLoss(model=model,
                frac=cfg.frac, 
                fc_layer_name = 'fc', 
                lambda_anchor=cfg.lambda_anchor)

    elif loss_name == 'AnchorHRegLoss':
        return AnchorHRegLoss(model=model,
                fc_layer_name ='fc', 
                frac=cfg.frac, 
                hreg_decay=cfg.hreg_decay,
                lambda_anchor=cfg.lambda_anchor,
                )
        
    elif loss_name == 'CenterLoss':
        return CenterLoss(model, 
                fc_layer_name='fc', 
                frac=cfg.frac, 
                lambda_center=cfg.lambda_center
                )

    elif loss_name == 'AnchorCenterLoss':
        return AnchorCenterLoss(model, 
                fc_layer_name='fc', 
                frac=cfg.frac, 
                lambda_anchor=cfg.lambda_anchor,
                lambda_center=cfg.lambda_center
                )


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
        exec("model." + ly + ".register_forward_hook(get_activation('" + ly + "', activation))")
        
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
