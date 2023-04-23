from ffNetwork import *
from utilities import *
from cnn import CNN
from tqdm import tqdm
import wandb
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval(model, criterion, dataloader):

    model.eval() # set model in evaluation mode
    vloss, vacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (images, labels) in enumerate(dataloader):

        ### Move data to device (ideally GPU)
        images      = images.to(DEVICE)
        labels    = labels.to(DEVICE)

        # makes sure that there are no gradients computed as we are not training the model now
        with torch.inference_mode(): 
            ### Forward Propagation
            logits  = model(images)
            ### Loss Calculation
            loss    = criterion(logits, labels)

        vloss   += loss.item()
        vacc    += torch.sum(torch.argmax(logits, dim= 1) == labels).item()/logits.shape[0]
        
        # Do you think we need loss.backward() and optimizer.step() here?

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))), 
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()
    
        ### Release memory
        del images, labels, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(dataloader)
    vacc    /= len(dataloader)

    return vloss, vacc

def backproptrain(model, dataloader, optimizer, criterion):

    model.train()
    tloss, tacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    
    for i, (images, labels) in enumerate(dataloader):
        ### Initialize Gradients
        optimizer.zero_grad()
        ### Move Data to Device (Ideally GPU)
        images      = images.to(DEVICE)
        labels    = labels.to(DEVICE)

        logits  = model(images)
          ### Loss Calculation
        loss    = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        ### Backward Propagation
        #loss.backward() 
        
        ### Gradient Descent
        #optimizer.step()       

        tloss   += loss.item()
        tacc    += torch.sum(torch.argmax(logits, dim= 1) == labels).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))), 
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del images, labels, logits
        torch.cuda.empty_cache()
  
    batch_bar.close()
    tloss   /= len(train_loader)
    tacc    /= len(train_loader)

    return tloss, tacc

# batch_bar = tqdm(total=config['epochs'], dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=3)

config={
    'epochs':100,
    'lr':0.001,
    'batch_size':64,
    'threshold':1.0,
}
train_loader, test_loader = get_loaders(config['batch_size'])
visualise_positive()
visualise_negative()
# wandb.login(key="95ecc4ba8c0966a365ccf643e67d08226e659d02") #API Key is in your wandb account, under settings (wandb.ai/settings)
# Create your wandb run
run = wandb.init(
    project="project",
    entity  = "automellon",
    name = "CNN-MNIST", ## Wandb creates random run names if you skip this field
    reinit = False, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    config = config ### Wandb Config for your run
)

model = CNN(10).to(DEVICE)
best_val_acc=0.0
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr']) #Defining Optimizer 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, threshold=0.01)
for epoch in range(config['epochs']):

    print("\nEpoch {}/{}".format(epoch+1, config['epochs']))

    curr_lr                 = float(optimizer.param_groups[0]['lr'])
    train_loss, train_acc   = backproptrain(model, train_loader, optimizer, criterion)
    val_loss, val_acc       = eval(model, criterion, test_loader)
    #call scheduler.step with validation accuracy here. also read the docs for it

    print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
    print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))

    ### Log metrics at each epoch in your run 
    # Optionally, you can log at each batch inside train/eval functions 
    # (explore wandb documentation/wandb recitation)
    wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss, 
               'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})
    if(val_acc>best_val_acc):
      best_val_acc=val_acc
      file_name =  "backnet "+str(epoch)
      checkpoint = {"model": model.state_dict()}
      # Write checkpoint as desired, e.g.,
      torch.save(checkpoint,file_name)
      wandb.save(file_name)
#   batch_bar.set_postfix(train_loss = "{:.04f}".format(float(train_l)),test_loss="{:.04f}".format(float(test_l)))
#   batch_bar.update()
# batch_bar.close()
run.finish()