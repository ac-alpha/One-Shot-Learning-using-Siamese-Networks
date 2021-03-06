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
if torch.cuda.is_available():
  net.cuda()
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
        labels = generateLabels(batch_size)
        if torch.cuda.is_available():
          img1 = img1.cuda()
          img2 = img2.cuda()
          labels = labels.cuda()
        output = net.forward(img1,img2)
        output = net.forward(img1,img2)
        
        loss = criterion(output, labels)
        current_loss = loss.item()
        loss_record.append(current_loss)
        loss.backward()
        optimizer.step()
        print("Epoch:%d Batch:%d Loss:%.5f Time Lapsed:%s"%(epoch,batch_idx+1,current_loss,time.time() - start_time))

import matplotlib.pyplot as plt
plt.plot(loss_record)
plt.show()

with torch.no_grad() :
    net = SiameseNet()
    net.load_state_dict(torch.load('siamese.pth'))
    net.eval()
    val_batch = getValBatch()
    img1 = val_batch[:,0,:,:,:].view(-1,1,105,105)
    img2 = val_batch[:,1,:,:,:].view(-1,1,105,105)
    if torch.cuda.is_available():
        net.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()
    output = net.forward(img1,img2)
    output = output >= 0.5
    correct = 0
    for i in range(len(val_batch)):
        if(i%2 == 0):
            if(output[i] == 1):
                correct += 1
        else:
            if(output[i] == 0):
                correct += 1
    print(correct/float(len(val_batch)))