import torch
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset

class LinearRegressionManual:
    def __init__(self, in_features):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    # Обучаем модель
    model = LinearRegressionManual(in_features=1)
    lr = 0.1
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
    
    model.save('linreg_manual.pth') 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from utils import make_classification_data, log_epoch, ClassificationDataset

def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

class LogisticRegressionManual:
    def __init__(self, in_features, n_classes=2):
        self.n_classes = n_classes
        self.w = torch.randn(in_features, n_classes, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(n_classes, dtype=torch.float32, requires_grad=False)

    def __call__(self, X):
        logits = X @ self.w + self.b
        return softmax(logits) if self.n_classes > 2 else torch.sigmoid(logits)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        if self.n_classes > 2:
            # Для многоклассовой классификации
            error = y_pred - y
            self.dw = (X.T @ error) / n
            self.db = error.mean(0)
        else:
            # Для бинарной классификации
            error = y_pred - y.unsqueeze(1)
            self.dw = (X.T @ error) / n
            self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b, 'n_classes': self.n_classes}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']
        self.n_classes = state.get('n_classes', 2)

def accuracy(y_pred, y_true):
    if y_pred.shape[1] > 1:  # Многоклассовая классификация
        preds = torch.argmax(y_pred, dim=1)
        return (preds == y_true).float().mean().item()
    else:  # Бинарная классификация
        preds = (y_pred > 0.5).float().squeeze()
        return (preds == y_true).float().mean().item()

def calculate_metrics(y_true, y_pred, n_classes=2):
    y_true_np = y_true.numpy()
    
    if n_classes > 2:
        y_pred_np = torch.argmax(y_pred, dim=1).numpy()
        precision = precision_score(y_true_np, y_pred_np, average='weighted')
        recall = recall_score(y_true_np, y_pred_np, average='weighted')
        f1 = f1_score(y_true_np, y_pred_np, average='weighted')
        roc_auc = roc_auc_score(y_true_np, y_pred.numpy(), multi_class='ovo', average='weighted')
    else:
        y_pred_np = (y_pred > 0.5).float().squeeze().numpy()
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        roc_auc = roc_auc_score(y_true_np, y_pred.squeeze().numpy())
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def plot_confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = range(len(torch.unique(y_true)))
    
    if y_pred.shape[1] > 1:
        y_pred_labels = torch.argmax(y_pred, dim=1)
    else:
        y_pred_labels = (y_pred > 0.5).float().squeeze()
    
    cm = confusion_matrix(y_true.numpy(), y_pred_labels.numpy())
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Пример для многоклассовой классификации
    n_classes = 3
    X, y = make_classification_data(n=200, n_classes=n_classes)
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    # Обучаем модель
    model = LogisticRegressionManual(in_features=2, n_classes=n_classes)
    lr = 0.1
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_targets = []
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            
            # Вычисляем loss (cross-entropy)
            if model.n_classes > 2:
                loss = -torch.sum(torch.log(y_pred + 1e-8) * batch_y) / batch_X.shape[0]
            else:
                loss = -(batch_y * torch.log(y_pred + 1e-8) + (1 - batch_y) * torch.log(1 - y_pred + 1e-8)).mean()
            
            acc = accuracy(y_pred, batch_y)
            
            total_loss += loss.item()
            total_acc += acc
            
            # Сохраняем предсказания и метки для метрик
            all_preds.append(y_pred.detach())
            all_targets.append(batch_y.detach())
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        # Вычисляем метрики
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = calculate_metrics(all_targets, all_preds, n_classes=model.n_classes)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
            print(f'Metrics - Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, '
                  f'F1: {metrics["f1"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}')
    
    model.save('logreg_manual.pth')
    
    # Визуализация confusion matrix
    with torch.no_grad():
        X_test, y_test = dataset[:][0], dataset[:][1]
        y_pred_test = model(X_test)
        plot_confusion_matrix(y_test, y_pred_test, classes=range(n_classes))