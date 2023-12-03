if __name__ == '__main__':
    import numpy as np
    from skimage import io
    import matplotlib
    import matplotlib.pyplot as plt
    import os
    from glob import glob
    import torch
    import torch.nn as nn
    from tqdm import tqdm
    from PIL import Image

    from torch.utils.data import DataLoader, Dataset
    from torch.autograd import Variable
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')


    # Global parameters

    # If USE_CUDA is True, computations will be done using the GPU (may not work in all systems)
    # This will make the calculations happen faster
    USE_CUDA = torch.cuda.is_available()

    DATASET_PATH = './data'

    BATCH_SIZE = 64 # Number of images that are used for calculating gradients at each step

    NUM_EPOCHS = 25 # Number of times we will go through all the training images. Do not go over 25

    LEARNING_RATE = 0.001 # Controls the step size
    MOMENTUM = 0.9 # Momentum for the gradient descent
    WEIGHT_DECAY = 0.0005
    # Create datasets and data loaders
    # Transformations

    from torchvision import datasets, models, transforms
    data_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train'), data_transforms)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


    test_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'test'), data_transforms)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_dataset.classes

    print('Dataloaders OK')
    test_loader

    #Print the corresponding label for the image

    random_image = train_dataset[13421][0].numpy().transpose((1, 2, 0))   
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    random_image = std * random_image + mean
    random_image = np.clip(random_image, 0, 1)
    print("Image label:", train_dataset[13421][1])
    plt.imshow(random_image)

    #Create the model class
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            #Same Padding = [(filter size - 1) / 2] (Same Padding--> input size = output size)
            self.cnn1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3,stride=1, padding=1)
            #The output size of each of the 4 feature maps is 
            #[(input_size - filter_size + 2(padding) / stride) +1] --> [(64-3+2(1)/1)+1] = 64 (padding type is same)
            self.batchnorm1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    
            #After max pooling, the output of each feature map is now 64/2 =32
            self.cnn2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
            #Output size of each of the 32 feature maps
            self.batchnorm2 = nn.BatchNorm2d(8)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            
            #After max pooling, the output of each feature map is 32/2 = 16
            #Flatten the feature maps. You have 8 feature maps, each of them is of size 16x16 --> 8*16*16 = 2048
            self.fc1 = nn.Linear(in_features=8*16*16, out_features=32)
            self.droput = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(in_features=32, out_features=10)
            
        def forward(self,x):
            out = self.cnn1(x)
            out = self.batchnorm1(out)
            out = self.relu(out)
            out = self.maxpool1(out)
            out = self.cnn2(out)
            out = self.batchnorm2(out)
            out = self.relu(out)
            out = self.maxpool2(out)
            
            #Now we have to flatten the output. This is where we apply the feed forward neural network as learned before! 
            #It will take the shape (batch_size, 2048)
            out = out.view(x.size(0), -1)
            
            #Then we forward through our fully connected layer 
            out = self.fc1(out)
            out = self.relu(out)
            #out = self.droput(out)
            out = self.fc2(out)
            return out
        
    # Create network
    model = CNN()
    if USE_CUDA:
        model = model.cuda()  
        
    print('Network OK')



    # Define criterion, optimizer, and scheduler

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Main loop
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    epochs = []

    for epoch in range(1, NUM_EPOCHS+1):
        print(f'\n\nRunning epoch {epoch} of {NUM_EPOCHS}...\n')
        epochs.append(epoch)

        #-------------------------Train-------------------------
        
        #Reset these below variables to 0 at the begining of every epoch
        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train()  # Put the network into training mode
        
        for i, (inputs, labels) in enumerate(train_loader):
        
            if USE_CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()        
                
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iter_loss += loss.item()  # Accumulate the loss
            optimizer.zero_grad() # Clear off the gradient in (w = w - gradient)
            loss.backward()   # Backpropagation 
            optimizer.step()  # Update the weights
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1
            
        scheduler.step()
            
        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((100 * correct / len(train_dataset)))   
        
        #-------------------------Test--------------------------
        
        correct = 0
        iterations = 0
        testing_loss = 0.0
        
        model.eval()  # Put the network into evaluation mode
        
        for i, (inputs, labels) in enumerate(test_loader):

            if USE_CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = model(inputs)     
            loss = criterion(outputs, labels) # Calculate the loss
            testing_loss += loss.item()
            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            
            iterations += 1

        # Record the Testing loss
        test_loss.append(testing_loss/iterations)
        # Record the Testing accuracy
        test_accuracy.append((100 * correct / len(test_dataset)))
    
        print(f'\nEpoch {epoch} validation results: Loss={test_loss[-1]} | Accuracy={test_accuracy[-1]}\n')

        # Plot and save
        plt.figure(figsize=(12, 8), num=1)
        plt.clf()
        plt.plot(epochs, train_loss, label='Train')
        plt.plot(epochs, test_loss, label='Test')
        plt.legend()
        plt.grid()
        plt.title('Cross entropy loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('outputs/01-loss-cnn.pdf')

        plt.figure(figsize=(12, 8), num=2)
        plt.clf()
        plt.plot(epochs, train_accuracy, label='Train')
        plt.plot(epochs, test_accuracy, label='Test')
        plt.legend()
        plt.grid()
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('outputs/02-accuracy-cnn.pdf')

    #Result
    print(f'Final train loss: {train_loss[-1]}')
    print(f'Final test loss: {test_loss[-1]}')
    print(f'Final train accuracy: {train_accuracy[-1]}')
    print(f'Final test accuracy: {test_accuracy[-1]}')