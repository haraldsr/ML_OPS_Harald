import numpy as np
import torch
import click
import matplotlib.pyplot as plt

from tqdm import trange
from data import MnistDataset
from model import MyAwesomeModel
import os

@click.group()
def cli():
    pass


@click.command()

@click.option("--data_path", default = 'corruptmnist', help = 'Path of data')
@click.option("--model_path", default = 'trained_model.pth', help= 'Path and or name of model to save')
@click.option("--lr", default=1e-3, help='Learning rate to use for training')
@click.option("--epochs", default=3, help='Number of epochs for training')

def train(data_path, model_path, lr, epochs):
    print("Training day and night")
    print(os.getcwd())
    # Create model
    model = MyAwesomeModel(784, 10, [256, 128, 64], drop_p=0.5)
    # Get data
    train_set = MnistDataset(dataset_dir=data_path, train=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    # Test data for validation to save best model
    test_set = MnistDataset(dataset_dir=data_path, train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    # Define Loss function & Optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loss needs to be high in order to ensure the first loss is better
    curr_best_loss = np.inf

    running_train_loss, running_test_loss = [], []
    #For Status bar
    with trange(epochs, unit="carrots") as pbar:
        #Epoch loop
        for e in pbar:
            pbar.set_description(f"Epoch {e}")
            
            ## Training Loop
            running_loss = 0
            
            # Set model to train mode
            model.train()

    #         for images, labels in trainloader:
    #             optimizer.zero_grad()
                
    #             # Flatten images into a 784 long vector
    #             images = images.view(images.shape[0], -1)

    #             #remember that log_softmax ie. not probabilities
    #             log_ps = model.forward(images)
    #             loss = criterion(log_ps, labels)
    #             loss.backward()
    #             optimizer.step()
                
    #             running_loss += loss.item()
    #         else:
    #             mean_train_loss = running_loss/len(trainloader)
    #             print(f'\nTrain Loss: {mean_train_loss}')
            
    #         running_train_loss.append(mean_train_loss)
    #         plt.show()
    #         ## Validation step
    #         mean_loss, _ = validation(model, testloader, criterion)
    #         running_test_loss.append(mean_loss)
    #         #set bar
    #         pbar.set_postfix(train_loss=mean_train_loss, valid_loss=mean_loss)

    #         # Save model if it is the best performing

    #         #if mean_loss < curr_best_loss:
    #         checkpoint = {'input_size': 784,
    #                         'output_size': 10,
    #                         'hidden_layers': [each.out_features for each in model.hidden_layers],
    #                         'state_dict': model.state_dict()}
            #import pickle
            #with open(model_path, 'wb') as file:
            #    pickle.dump(checkpoint, file)
            #torch.save(checkpoint, model_path)
    print("hest")
    #gammelt gem plot
    #plt.plot(range(epochs), running_train_loss)
    #plt.plot(range(epochs), running_test_loss)
    #plt.show()
    
@click.command()
@click.option("--model_path", default = 'model/trained_model.pth', help= 'Path and or name of model to save')
@click.option("--data_path", default = 'corruptmnist', help = 'Path of data')
def evaluate(model_path, data_path):
    # Define model
    model = load_checkpoint(model_path)
    
    # Load test data    
    test_set = MnistDataset(dataset_dir=data_path, train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    validation(model, testloader)

def load_checkpoint(model_checkpoint):
    # From S1 exercise 6
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel(checkpoint['input_size'],
                      checkpoint['output_size'],
                      checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def validation(model, testloader, criterion=torch.nn.NLLLoss()):
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        for images, labels in testloader:

            # Flatten images into a 784 long vector
            images = images.view(images.shape[0], -1)

            output = model.forward(images)

            #Using NLLLoss function
            test_loss += criterion(output, labels).item()

            # Recall output of model is log_softmax, thus exp must be taken in order to obtain probabilities
            ps = torch.exp(output)

            # Get the predicted class
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    mean_loss = test_loss/len(testloader)
    mean_acc = accuracy/len(testloader)
    print(f'Mean Test Loss: {mean_loss}')
    print(f'Mean Test Accuracy: {mean_acc*100}%\n')

    return mean_loss, mean_acc


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()