
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_graphs(data):
    """تعیین تعداد گراف‌ها در یک شی Data"""
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.batch.max().item() + 1

def train_multiple_epochs(train_graphs, val_graphs, test_graphs, model, adj, fold, max_epochs=50):
    """
    آموزش مدل برای چند ایپوک و ارزیابی روی داده‌های تست.
    
    پارامترها:
        train_graphs: لیست گراف‌های آموزشی.
        val_graphs: لیست گراف‌های اعتبارسنجی.
        test_graphs: لیست گراف‌های تست.
        model: مدل گراف (instance of gGATLDA)
        adj: ماتریس adjacency مربوط به گراف اصلی.
        fold: شماره فولد (برای چاپ پیام‌های لاگ).
        max_epochs: تعداد ایپوک‌های آموزش.
        
    خروجی:
        دیکشنری شامل معیارهای ارزیابی نهایی و لیست خطاهای هر ایپوک.
    """
    print(f"Starting training for fold {fold}...")
    LR = 0.001
    batch_size = 16
    
    train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    fold_losses = []
    epoch_accuracies = []
    
    for epoch in range(1, max_epochs + 1):
        total_loss = 0
        correct_train = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y.view(-1).long())
            loss.backward()
            total_loss += loss.item() * num_graphs(data)
            optimizer.step()
            
            pred = out.argmax(dim=1)
            correct_train += (pred == data.y).sum().item()
        
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct_train / len(train_loader.dataset)
        
        # ارزیابی روی داده‌های اعتبارسنجی
        val_loss = 0
        correct_val = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = F.cross_entropy(out, data.y.view(-1).long())
                val_loss += loss.item() * num_graphs(data)
                pred = out.argmax(dim=1)
                correct_val += (pred == data.y).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / len(val_loader.dataset)
        
        fold_losses.append((train_loss, val_loss))
        epoch_accuracies.append((train_accuracy, val_accuracy))
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Train Acc: {train_accuracy:.5f}, Val Acc: {val_accuracy:.5f}')
    
    # ارزیابی نهایی روی داده‌های تست
    test_loss = 0
    correct_test = 0
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = F.cross_entropy(out, data.y.view(-1).long())
            test_loss += loss.item() * num_graphs(data)
            pred_prob = torch.softmax(out, dim=1)[:, 1]
            predictions.extend(pred_prob.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    predict_labels = (np.array(predictions) >= 0.5).astype(int)
    
    # محاسبه معیارهای ارزیابی
    fpr, tpr, _ = metrics.roc_curve(np.array(labels), np.array(predictions), pos_label=1)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(np.array(labels), np.array(predictions), pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    aupr_score = metrics.auc(recall_curve, precision_curve)
    f1 = metrics.f1_score(np.array(labels), predict_labels)
    accuracy = metrics.accuracy_score(np.array(labels), predict_labels)
    recall = metrics.recall_score(np.array(labels), predict_labels)
    precision = metrics.precision_score(np.array(labels), predict_labels)
    mcc = metrics.matthews_corrcoef(np.array(labels), predict_labels)
    
    print(f"Fold {fold} Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, Test Loss={test_loss:.4f}")
    
    return {
        'test_auc': auc_score,
        'f1': f1,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'mcc': mcc,
        'aupr_score': aupr_score,
        'test_loss': test_loss,
        'fold_losses': fold_losses,
        'epoch_accuracies': epoch_accuracies,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'truth': np.array(labels),
        'predictions': np.array(predictions)
    }
