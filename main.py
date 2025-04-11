

import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from preprocessing import load_data
from training import train_multiple_epochs
from utils import MyDataset, plot_roc_pr_curves

def main():
    parser = argparse.ArgumentParser(description="Predicting lncRNA-disease associations using GNN")
    parser.add_argument('--dataset', help='Dataset name (folder name under raw_data)', required=True)
    parser.add_argument('--num-gcn-layers', type=int, default=2, help='Number of GCN layers')
    args = parser.parse_args()
    
    # تنظیم بذر تصادفی برای قابلیت تکرار
    seed = 2341
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('Training...')
    # بارگذاری و پیش‌پردازش داده‌ها
    u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset)
    
    # تبدیل ماتریس شبکه به تانسور PyTorch
    adj = torch.tensor(net)
    all_indices = (u_indices, v_indices)
    
    # استخراج زیرگراف‌ها (این تابع در utils.py تعریف شده و از توابع موازی استفاده می‌کند)
    from utils import extracting_subgraphs  # وارد کردن تابع استخراج زیرگراف‌ها
    all_graphs = extracting_subgraphs(net, all_indices, labels, h=1, u_features=u_features, v_features=v_features, max_node_label=3)
    
    # ساخت دیتاست از گراف‌ها
    mydataset = MyDataset(all_graphs, root=f'data/{args.dataset}')
    
    # تنظیم اعتبارسنجی متقابل ۵-fold
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    labels_np = np.array([d.y.item() for d in mydataset])
    
    fold_metrics = []
    all_train_losses = []
    all_val_losses = []
    
    # برای هر فولد از داده‌های CV
    for fold, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
        print('*' * 25, f'Fold {fold + 1}', '*' * 25)
        
        # وارد کردن مدل
        from models import gGATLDA
        in_features = mydataset[0].x.shape[1]
        model = gGATLDA(in_features=in_features, num_gcn_layers=args.num_gcn_layers)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # تقسیم‌بندی داده‌ها
        train_val_data = [mydataset[i] for i in train_val_index]
        test_data = [mydataset[i] for i in test_index]
        
        # تقسیم‌یابی داخلی (برای اعتبارسنجی) از داده‌های train_val
        from sklearn.model_selection import StratifiedKFold
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_index, val_index = next(skf_inner.split(np.zeros(len(train_val_data)), [d.y.item() for d in train_val_data]))
        train_data = [train_val_data[i] for i in train_index]
        val_data = [train_val_data[i] for i in val_index]
        
        # آموزش و ارزیابی مدل برای فولد جاری
        fold_result = train_multiple_epochs(train_data, val_data, test_data, model, adj, fold + 1, max_epochs=50)
        fold_metrics.append(fold_result)
        
        # جمع‌آوری خطاهای هر ایپوک
        train_loss, val_loss = zip(*fold_result['fold_losses'])
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        
        print(f"Metrics for Fold {fold + 1}:")
        for key, value in fold_result.items():
            if key not in ["fold_losses", "epoch_accuracies", "fpr", "tpr", "precision_curve", "recall_curve", "truth", "predictions"]:
                print(f"{key}: {value:.4f}")
    
    # میانگین خطاها در سطح فولدها
    mean_train_losses = np.mean(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    
    # رسم نمودار میانگین خطا
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mean_train_losses) + 1), mean_train_losses, label='Training Loss')
    plt.plot(range(1, len(mean_val_losses) + 1), mean_val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs (Averaged Across Folds)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # محاسبه میانگین و واریانس معیارها
    metric_keys = [key for key in fold_metrics[0].keys() if key not in ["fold_losses", "epoch_accuracies", "fpr", "tpr", "precision_curve", "recall_curve", "truth", "predictions"]]
    mean_metrics = { key: np.mean([fold_metric[key] for fold_metric in fold_metrics]) for key in metric_keys }
    var_metrics = { key: np.var([fold_metric[key] for fold_metric in fold_metrics]) for key in metric_keys }
    
    print("\nMean Metrics Across 5 Folds:")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nVariance of Metrics Across 5 Folds:")
    for key, value in var_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # رسم نمودارهای ROC و PR
    plot_roc_pr_curves(fold_metrics, k_folds=K)

if __name__ == '__main__':
    main()
