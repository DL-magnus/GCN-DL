import torch
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_max_pool
from torch_scatter import scatter_mean, scatter_max 
import matplotlib.pyplot as plt
from torch_geometric.utils import dropout_adj

#Use wandb to optimise and log performance of system. 
# weights and biases

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 16, cached=False)
        self.conv2 = GCNConv(16, 32, cached=False)
        self.conv3 = GCNConv(32, 64, cached=False)
        self.conv4 = GCNConv(64, 64, cached=False)
        self.linear1 = torch.nn.Linear(64, 10)

    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.linear1(x)
        x = scatter_mean(x, batch, dim=0)  
        
        return F.log_softmax(x, dim=1)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_losses = []
val_losses = []
val_lower = 0.5

count = -1
for epoch in range(300):
    running_loss = 0.0
    val_running_loss = 0.0
    model.train()
    for i, batch in enumerate(train_loader, 0):
        count += 1
        data = batch.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    mean_loss = running_loss/(i+1)
    train_losses.append((count, mean_loss))
    print('Epoch:', epoch)
    print('Train loss:', mean_loss)
        
    model.eval()
    with torch.no_grad():
        for j, batch_val in enumerate(val_loader, 0):
            data_val = batch_val.to(device)
            out_val = model(data_val)
            val_loss = F.nll_loss(out_val, data_val.y)
            val_running_loss += val_loss.item()
        val_mean_loss = val_running_loss/(j+1)
        val_losses.append((count, val_mean_loss))
        print('Validation loss:', val_mean_loss)
        
    if val_mean_loss <= val_lower:
        break
              
print('Finished Training')

list1, list2 = zip(*train_losses)
list3, list4 = zip(*val_losses)

plt.plot(list1, list2, list3, list4)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(('Train', 'Validation'))
#plt.savefig('C:/Users/umaer/OneDrive/Documents/PhD/Code/Results/train_curve6.png')
