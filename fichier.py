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


    # Paramètres globaux

    # Cela accélérera les calculs
    USE_CUDA = torch.cuda.is_available()
    DATASET_PATH = './data'
    BATCH_SIZE = 64 # Nombre d'images utilisées pour calculer les gradients à chaque étape
    NUM_EPOCHS = 25 # Nombre de fois où nous parcourons toutes les images d'entraînement.
    LEARNING_RATE = 0.001 # Contrôle de la taille du pas
    MOMENTUM = 0.9 # Momentum pour la descente de gradient
    WEIGHT_DECAY = 0.0005
    # Créer des ensembles de données et des chargeurs de données
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

    print('Chargeurs de données prêts')
    test_loader

    # Imprimer l'étiquette correspondante pour l'image

    random_image = train_dataset[13421][0].numpy().transpose((1, 2, 0))   
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    random_image = std * random_image + mean
    random_image = np.clip(random_image, 0, 1)
    print("Étiquette de l'image:", train_dataset[13421][1])
    plt.imshow(random_image)

    # Créer la classe du modèle
    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            #Remplissage identique = [(taille du filtre - 1) / 2] (Remplissage identique--> taille d'entrée = taille de sortie)
            self.cnn1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3,stride=1, padding=1)
            #La taille de sortie de chacune des 4 cartes de caractéristiques est 
            #[(taille_entrée - taille_filtre + 2(remplissage) / stride) +1] --> [(64-3+2(1)/1)+1] = 64 (type de remplissage est identique)
            self.batchnorm1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    
            #Après le pooling maximal, la sortie de chaque carte de caractéristiques est maintenant de 64/2 =32
            self.cnn2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
            #Taille de sortie de chacune des 32 cartes de caractéristiques
            self.batchnorm2 = nn.BatchNorm2d(8)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            
            #Après le pooling maximal, la sortie de chaque carte de caractéristiques est de 32/2 = 16
            #Aplatir les cartes de caractéristiques. Vous avez 8 cartes de caractéristiques, chacune d'elles est de taille 16x16 --> 8*16*16 = 2048
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
            
            #Maintenant, nous devons aplatir la sortie. C'est là que nous appliquons le réseau de neurones à propagation avant comme appris précédemment ! 
            #Il prendra la forme (batch_size, 2048)
            out = out.view(x.size(0), -1)
            
            #Ensuite, nous avançons à travers notre couche entièrement connectée 
            out = self.fc1(out)
            out = self.relu(out)
            #out = self.droput(out)
            out = self.fc2(out)
            return out
        
    # Créer le réseau
    model = CNN()
    if USE_CUDA:
        model = model.cuda()  
        
    print('Réseau prêt')



    # Définir le critère, l'optimiseur et le planificateur

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Boucle principale
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    epochs = []

    for epoch in range(1, NUM_EPOCHS+1):
        print(f'\n\nExécution de l\'époque {epoch} sur {NUM_EPOCHS}...\n')
        epochs.append(epoch)

        #-------------------------Entraînement-------------------------
        
        #Réinitialiser ces variables ci-dessous à 0 au début de chaque époque
        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train()  # Mettre le réseau en mode d'entraînement
        
        for i, (inputs, labels) in enumerate(train_loader):
        
            if USE_CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()        
                
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            iter_loss += loss.item()  # Accumuler la perte
            optimizer.zero_grad() # Effacer le gradient dans (w = w - gradient)
            loss.backward()   # Rétropropagation 
            optimizer.step()  # Mettre à jour les poids
            
            # Enregistrer les prédictions correctes pour les données d'entraînement 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1
            
        scheduler.step()
            
        # Enregistrer la perte d'entraînement
        train_loss.append(iter_loss/iterations)
        # Enregistrer la précision d'entraînement
        train_accuracy.append((100 * correct / len(train_dataset)))   
        
        #-------------------------Test--------------------------
        
        correct = 0
        iterations = 0
        testing_loss = 0.0
        
        model.eval()  # Mettre le réseau en mode d'évaluation
        
        for i, (inputs, labels) in enumerate(test_loader):

            if USE_CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = model(inputs)     
            loss = criterion(outputs, labels) # Calculer la perte
            testing_loss += loss.item()
            # Enregistrer les prédictions correctes pour les données de test
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            
            iterations += 1

        # Enregistrer la perte de test
        test_loss.append(testing_loss/iterations)
        # Enregistrer la précision de test
        test_accuracy.append((100 * correct / len(test_dataset)))
    
        print(f'\nRésultats de validation pour l\'époque {epoch}: Perte={test_loss[-1]} | Précision={test_accuracy[-1]}\n')

        # Tracer et enregistrer
        plt.figure(figsize=(12, 8), num=1)
        plt.clf()
        plt.plot(epochs, train_loss, label='Entraînement')
        plt.plot(epochs, test_loss, label='Test')
        plt.legend()
        plt.grid()
        plt.title('Perte d\'entropie croisée')
        plt.xlabel('Époque')
        plt.ylabel('Perte')
        plt.savefig('outputs/01-loss-cnn.pdf')
        plt.show()

        plt.figure(figsize=(12, 8), num=2)
        plt.clf()
        plt.plot(epochs, train_accuracy, label='Entraînement')
        plt.plot(epochs, test_accuracy, label='Test')
        plt.legend()
        plt.grid()
        plt.title('Précision')
        plt.xlabel('Époque')
        plt.ylabel('Précision')
        plt.savefig('outputs/02-accuracy-cnn.pdf')
        plt.show()

    #Résultat
    print(f'Perte d\'entraînement finale: {train_loss[-1]}')
    print(f'Perte de test finale: {test_loss[-1]}')
    print(f'Précision d\'entraînement finale: {train_accuracy[-1]}')
    print(f'Précision de test finale: {test_accuracy[-1]}')
