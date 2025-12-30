import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    train_loss, train_pos_acc, train_neg_acc = 0.0, 0.0, 0.0
    
    model.train()
    for batch, X in tqdm(enumerate(dataloader)):
        
        X, y = X[:, :5].to(device), X[:, 5:].to(device)
        B, T, _, H, W = X.shape
        y_preds = model(X)
        y_preds_labels = torch.round(torch.sigmoid(y_preds)).float()
        y = y.view(B * T, 1, H, W)
        y_preds = y_preds.view(B * T, 1, H, W)
        y_preds_labels = y_preds_labels.view(B * T, 1, H, W)
        loss = loss_fn(y_preds, y)
        
        with torch.no_grad():
            pos_acc = torch.logical_and(y == 1, y_preds_labels == 1).sum()/(y == 1).sum()
            neg_acc = torch.logical_and(y == 0, y_preds_labels == 0).sum()/(y == 0).sum()
    
            train_loss += loss.item()
            train_pos_acc += pos_acc
            train_neg_acc += neg_acc
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= 1.0)
        optimizer.step()

    train_loss /= len(dataloader)
    train_pos_acc /= len(dataloader)
    train_neg_acc /= len(dataloader)

    print(f"Train Loss: {train_loss:.5f} | Train Recall: {train_pos_acc:.4f} | Train Specificity: {train_neg_acc:.4f}")

    return train_loss, train_pos_acc, train_neg_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, 
              device: torch.device):

    test_loss, test_pos_acc, test_neg_acc = 0.0, 0.0, 0.0
    
    model.eval()
    with torch.inference_mode():
        for batch, X in tqdm(enumerate(dataloader)):
            
            X, y = X[:, :5].to(device), X[:, 5:].to(device)
            
            B, T, _, H, W = X.shape
            y_preds = model(X)
            y_preds_labels = torch.round(torch.sigmoid(y_preds)).float()
            
            y = y.view(B * T, 1, H, W)
            y_preds = y_preds.view(B * T, 1, H, W)
            y_preds_labels = y_preds_labels.view(B * T, 1, H, W)
            loss = loss_fn(y_preds, y)
            
            pos_acc = torch.logical_and(y == 1, y_preds_labels == 1).sum()/(y == 1).sum()
            neg_acc = torch.logical_and(y == 0, y_preds_labels == 0).sum()/(y == 0).sum()
            test_loss += loss.item()
            test_pos_acc += pos_acc
            test_neg_acc += neg_acc
        
        test_loss /= len(dataloader)
        test_pos_acc /= len(dataloader)
        test_neg_acc /= len(dataloader)
        
        print(f"Test Loss: {test_loss:.5f} | Test Recall: {test_pos_acc:.4f} | Test Specificity: {test_neg_acc:.4f}")

    return test_loss, test_pos_acc, test_neg_acc

def train_MovingMNIST(model: torch.nn.Module,
                       train_dataloader: torch.utils.data.DataLoader,
                       test_dataloader: torch.utils.data.DataLoader,
                       loss_fn: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: torch.device,
                       epochs: int):

    results = {"train_loss": [],
               "train_pos_acc": [],
               "train_neg_acc": [],
               "test_loss": [],
               "test_pos_acc": [],
               "test_neg_acc": []}
    model.to(device)
    
    for epochs in tqdm(range(epochs)):
        train_loss, train_pos_acc, train_neg_acc = train_step(model= model,
                                                              dataloader= train_dataloader,
                                                              loss_fn= loss_fn,
                                                              optimizer= optimizer,
                                                              device= device)

        test_loss, test_pos_acc, test_neg_acc = test_step(model= model,
                                                          dataloader= test_dataloader,
                                                          loss_fn= loss_fn,
                                                          device= device)

        results["train_loss"].append(train_loss)
        results["train_pos_acc"].append(train_pos_acc)
        results["train_neg_acc"].append(train_neg_acc)
        results["test_loss"].append(test_loss)
        results["test_pos_acc"].append(test_pos_acc)
        results["test_neg_acc"].append(test_neg_acc)

    return results
