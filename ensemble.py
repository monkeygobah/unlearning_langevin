import torch
from make_dataloaders import *
from tqdm import tqdm
import time

def run_exps(args, testset, trainset, train_remain_loader, finetune=False, frozen=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    good_r = torch.load(args.good_remain, map_location=device)
    good_f = torch.load(args.good_forget, map_location=device)
    print(train_remain_loader.batch_size)

    if frozen:
        for param in good_r.parameters():
            param.requires_grad = False

        for param in good_r.module.classifier.parameters():
            param.requires_grad = True

    test_forget_loader, test_remain_loader = get_forget_loader(testset, args.forget_class)

    fc_from_good_f = good_f.module.classifier
    good_r.module.classifier = fc_from_good_f

    epochs = 0
    start=time.time()
    if finetune:

        optimizer = torch.optim.Adam(good_r.parameters(), lr=1e-4)  
        criterion = torch.nn.CrossEntropyLoss()  

        epochs = 5  
        good_r.train() 

        for epoch in range(epochs):
            running_loss = 0.0

            for step, (batch_x, batch_y) in enumerate(tqdm(train_remain_loader)):

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = good_r(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_remain_loader)}')

    end=time.time()
    print(f'Time taken: {end-start}')

    remain_accuracy = evaluate(good_r, test_remain_loader,device)
    forget_accuracy = evaluate(good_r, test_forget_loader,device)

    print(f'Remain Set Accuracy with finetune:{finetune} for {epochs}: {remain_accuracy}')
    print(f'Forget Set Accuracy with finetune:{finetune} for {epochs}: {forget_accuracy}')


def evaluate(model, dataloader,device):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total