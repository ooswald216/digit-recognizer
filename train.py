# -*- coding:utf-8 -*-
"""
作者：ooswald216
日期：2022年03月14日
"""
from model import CNNModel
import torch.nn as nn
import torch
from torch.autograd import Variable
from data import num_epochs,train_loader,test_loader
from matplotlib import pyplot as plt

model = CNNModel()
error = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()

        optimizer.step()
        count += 1

        if count % 50 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                test = Variable(images.view(100, 1, 28, 28))
                outputs = model(test)
                #[1]找最大值对应的标签
                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 500 == 0:
            print('Iteration: {} Loss: {} Accuracy:{} %'.format(count,loss.data,accuracy))

save_info = {
    "iter_num": num_epochs,
    "optimizer": optimizer.state_dict(),
    "model": model.state_dict(),
}
torch.save(save_info, "./model/CNNModel.pt")

# plt.plot(iteration_list,loss_list)
# plt.xlabel("Number of iteration")
# plt.ylabel("Loss")
# plt.title("CNN: Loss vs Number of iteration")
# plt.show()
#
# # visualization accuracy
# plt.plot(iteration_list,accuracy_list,color = "red")
# plt.xlabel("Number of iteration")
# plt.ylabel("Accuracy")
# plt.title("CNN: Accuracy vs Number of iteration")
# plt.show()