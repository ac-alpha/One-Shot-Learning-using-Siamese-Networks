import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import data_loader
import model
import time

start_time = time.time()

def initWeights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.2)
        m.bias.data.normal_(mean = 0.5, std = 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.normal_(mean = 0.5, std = 0.01)

def generateLabels(batch_size):
    labels = torch.zeros(batch_size)
    for i in range(batch_size):
        if i%2==0:
            labels[i]=1
    return labels

print(generateLabels(128))

net = model.SiameseNet()
net.apply(initWeights)

optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay = 0.1)

criterion = nn.BCELoss()

print("----Loading Data-----")

train_batches = data_loader.getTrainBatches(n = 30000, batch_size = 64)

print("Data loaded in %s seconds"%(time.time()-start_time))

start_time = time.time()
print("----Training-----")

loss_record = []

for epoch in range(1,101):
     
    for batch_idx,mini_batch in enumerate(train_batches):

        optimizer.zero_grad()

        batch_size = mini_batch.size()[0]
        img1 = mini_batch[:,0,:,:,:].view(-1,1,105,105)
        img2 = mini_batch[:,1,:,:,:].view(-1,1,105,105)
        output = net.forward(img1,img2)
        labels = generateLabels(batch_size)
        loss = criterion(output, labels)
        current_loss = loss.item()
        loss_record.append(current_loss)
        loss.backward()
        optimizer.step()
        print("Epoch:%d Batch:%d Loss:%.5f Time Lapsed:%s"%(epoch,batch_idx+1,current_loss,time.time() - start_time))

