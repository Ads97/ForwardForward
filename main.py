from ffNetwork import *
from utilities import *
from tqdm import tqdm
from cnnff import CNNFF
import wandb
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config={
    'epochs':100,
    'lr':0.1,
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
    name = "FF-MNIST", ## Wandb creates random run names if you skip this field
    reinit = False, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    config = config ### Wandb Config for your run
)
net = CNNFF([(1, 64, 2, 2),(64, 128, 2, 2),(128, 256, 2, 2),(256, 512, 2, 2)], config).to(DEVICE)

# batch_bar = tqdm(total=config['epochs'], dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=3)
for epoch in range(config['epochs']):
  train_err=[] 
  train_loss = 0.0
  for x,y in tqdm(train_loader):
    x,y = x.cuda(),y.cuda()
    x_pos = overlay_y_on_x(x, y) 
    y_neg = get_y_neg(y)
    x_neg = overlay_y_on_x(x, y_neg)

    train_loss += net.train(x_pos, x_neg)
    train_err.append(net.predict(x).eq(y).float().mean().item())
  
  train_e = sum(train_err)/len(train_loader)
  train_loss /=len(train_loader)
  print("train accuracy",train_e)
  print("train loss",train_loss)
  test_err=[]
  for x_te,y_te in test_loader:
    x_te, y_te = x_te.cuda(), y_te.cuda()
    test_err.append(net.predict(x_te).eq(y_te).float().mean().item())
  
  test_e = sum(test_err)/len(test_loader)
  # scheduler.step(test_e)
  print("test accuracy",test_e)
  wandb.log({"train_accuracy":train_e, 'train_loss':train_loss,'validation_accuracy': test_e})
  # print('test error:', sum(test_loss)/len(test_loader))
#   batch_bar.set_postfix(train_loss = "{:.04f}".format(float(train_l)),test_loss="{:.04f}".format(float(test_l)))
#   batch_bar.update()
# batch_bar.close()
run.finish()