
import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_1samp

from preprocessing import load_data
from training import train_multiple_epochs
from utils import MyDataset, plot_roc_pr_curves, plot_confusion_matrix, plot_tsne, extracting_subgraphs

# برای گزارش FLOPs از thop استفاده می‌کنیم (پیشنهاد می‌شود pip install thop را اجرا کنید)
try:
    from thop import profile
except ImportError:
    print("thop library not found. Please install it using 'pip install thop'")
    profile = None

def plot_tsne_comparison(original_embeddings, model_embeddings, labels, 
                         title_original="Original", title_model="My Model"):
    from sklearn.manifold import TSNE
    tsne_orig = TSNE(n_components=2, random_state=2341)
    orig_2d = tsne_orig.fit_transform(original_embeddings)
    
    tsne_model = TSNE(n_components=2, random_state=2341)
    model_2d = tsne_model.fit_transform(model_embeddings)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    for lb in np.unique(labels):
        idx = labels==lb
        plt.scatter(orig_2d[idx,0], orig_2d[idx,1], label=f'Class {lb}', alpha=0.7)
    plt.title(f"t-SNE - {title_original}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    for lb in np.unique(labels):
        idx = labels==lb
        plt.scatter(model_2d[idx,0], model_2d[idx,1], label=f'Class {lb}', alpha=0.7)
    plt.title(f"t-SNE - {title_model}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Predicting lncRNA-disease associations using GNN")
    parser.add_argument('--dataset', help='Dataset name (folder under raw_data)', required=True)
    parser.add_argument('--num-gcn-layers', type=int, default=2, help='Number of GCN layers')
    args = parser.parse_args()

    seed = 2341
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # چاپ تنظیمات و سخت‌افزار
    print("=== Training Plan ===")
    print(" - 5-Fold CV with nested 5-Fold for internal validation")
    print(" - Max Epochs per fold: 50")
    print(" - Batch size: 16")
    print(" - Learning Rate: 0.001")
    print(f" - Number of GCN layers: {args.num_gcn_layers}")
    print(f" - Random Seed: {seed}")
    if torch.cuda.is_available():
        print(f"Hardware: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Hardware: CPU")
    print("=====================")

    print("Training...")
    # بارگذاری داده‌ها
    u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset)
    
    adj = torch.tensor(net)
    all_indices = (u_indices, v_indices)
    
    all_graphs = extracting_subgraphs(net, all_indices, labels, h=1,
                                      u_features=u_features, v_features=v_features,
                                      max_node_label=3)
    
    mydataset = MyDataset(all_graphs, root=f'data/{args.dataset}')
    
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    labels_np = np.array([d.y.item() for d in mydataset])
    
    fold_metrics = []
    all_train_losses = []
    all_val_losses = []
    test_accuracies = []
    
    overall_original_feats = []  # برای t-SNE ویژگی‌های اولیه
    overall_model_feats = []     # برای t-SNE ویژگی‌های مدل
    overall_labels = []
    
    # اندازه‌گیری زمان کل آموزش
    total_training_time = 0.0
    
    for fold, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
        print("\n" + "*"*25, f"Fold {fold+1}", "*"*25)
        
        in_features = mydataset[0].x.shape[1]
        from models import gGATLDA
        model = gGATLDA(in_features=in_features, num_gcn_layers=args.num_gcn_layers)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # محاسبه FLOPs با استفاده از thop (در صورت موجود بودن)
        if profile is not None:
            model.eval()
            sample_data = mydataset[0].to(device)
            try:
                flops, params = profile(model, inputs=(sample_data,))
                print(f"FLOPs: {flops/1e9:.4f} GFLOPs, Parameters: {params}")
            except Exception as e:
                print("Error computing FLOPs:", e)
        else:
            print("FLOPs measurement not performed (thop not available).")
        
        # چاپ پارامترهای مدل
        print(f"\nModel parameters for fold {fold+1}:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} -> shape={tuple(param.data.shape)}, num_params={param.numel()}")
        num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Complexity: {num_params_total} trainable parameters\n")
        
        train_val_data = [mydataset[i] for i in train_val_index]
        test_data = [mydataset[i] for i in test_index]
        
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_index, val_index = next(skf_inner.split(np.zeros(len(train_val_data)), [d.y.item() for d in train_val_data]))
        train_data = [train_val_data[i] for i in train_index]
        val_data = [train_val_data[i] for i in val_index]
        
        # اندازه‌گیری زمان آموزش فولد
        start_time = time.time()
        fold_result = train_multiple_epochs(train_data, val_data, test_data, model, adj, fold+1, max_epochs=50)
        fold_training_time = time.time() - start_time
        total_training_time += fold_training_time
        print(f"Training time for Fold {fold+1}: {fold_training_time:.2f} seconds")
        
        fold_metrics.append(fold_result)
        train_loss, val_loss = zip(*fold_result['fold_losses'])
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        test_accuracies.append(fold_result['accuracy'])
        
        # استخراج ویژگی‌های تست جهت t-SNE
        from torch_geometric.data import DataLoader
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        
        original_feats_list = []
        model_feats_list = []
        label_list = []
        
        # اندازه‌گیری زمان استنتاج
        inference_times = []
        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                start_inf = time.time()
                # استخراج embedding از مدل (با استفاده از متد get_embedding، که باید در مدل تعریف شده باشد)
                emb_model = model.get_embedding(batch_data)
                inf_time = time.time() - start_inf
                inference_times.append(inf_time)
                
                # محاسبه میانگین ویژگی‌های اولیه (original) برای هر گراف در batch
                n_graphs = batch_data.batch.max().item() + 1
                for i in range(n_graphs):
                    mask = (batch_data.batch == i)
                    x_i = batch_data.x[mask]
                    mean_i = x_i.mean(dim=0).cpu().numpy()
                    original_feats_list.append(mean_i)
                
                model_feats_list.extend(emb_model.cpu().numpy())
                label_list.extend(batch_data.y.view(-1).cpu().numpy())
        
        avg_inference_time = np.mean(inference_times)
        print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
        
        overall_original_feats.append(np.array(original_feats_list))
        overall_model_feats.append(np.array(model_feats_list))
        overall_labels.append(np.array(label_list))
    
    print("\nOverall Training Time for 5 folds: {:.2f} seconds".format(total_training_time))
    
    mean_train_losses = np.mean(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(mean_train_losses)+1), mean_train_losses, label='Training Loss')
    plt.plot(range(1, len(mean_val_losses)+1), mean_val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs (Averaged Across Folds)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    metric_keys = [k for k in fold_metrics[0].keys() if k not in ["fold_losses", "epoch_accuracies", "fpr", "tpr", 
                                                                    "precision_curve", "recall_curve", "truth", 
                                                                    "predictions", "confusion_matrix", "test_embeddings"]]
    mean_metrics = { key: np.mean([fm[key] for fm in fold_metrics]) for key in metric_keys }
    var_metrics = { key: np.var([fm[key] for fm in fold_metrics]) for key in metric_keys }
    print("\nMean Metrics Across 5 Folds:")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nVariance of Metrics Across 5 Folds:")
    for key, value in var_metrics.items():
        print(f"{key}: {value:.4f}")
    
    t_stat, p_value = ttest_1samp(test_accuracies, 0.5)
    print(f"\nT-test for test accuracies (chance level = 0.5): t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    plot_roc_pr_curves(fold_metrics, k_folds=K)
    
    overall_cm = np.sum([fm['confusion_matrix'] for fm in fold_metrics], axis=0)
    print("Overall Confusion Matrix:")
    print(overall_cm)
    plot_confusion_matrix(overall_cm, title="Overall Confusion Matrix")
    
    overall_original_feats = np.concatenate(overall_original_feats, axis=0)
    overall_model_feats = np.concatenate(overall_model_feats, axis=0)
    overall_labels = np.concatenate(overall_labels, axis=0)
    
    print("Generating side-by-side t-SNE (Original vs My Model) ...")
    plot_tsne_comparison(
        original_embeddings=overall_original_feats,
        model_embeddings=overall_model_feats,
        labels=overall_labels,
        title_original="lncRNA-disease pair feature (Original)",
        title_model="lncRNA-disease pair feature (My Model)"
    )

if __name__ == '__main__':
    main()






# # ************************8
# import argparse
# import random
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
# from sklearn.manifold import TSNE
# from scipy.stats import ttest_rel

# from preprocessing import load_data
# from training import train_multiple_epochs
# from utils import MyDataset, plot_roc_pr_curves

# def main():
#     parser = argparse.ArgumentParser(description="Predicting lncRNA-disease associations using GNN")
#     parser.add_argument('--dataset', help='Dataset name (folder name under raw_data)', required=True)
#     parser.add_argument('--num-gcn-layers', type=int, default=2, help='Number of GCN layers')
#     args = parser.parse_args()

#     seed = 2341
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     print('Training...')
#     u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset)
#     adj = torch.tensor(net)
#     all_indices = (u_indices, v_indices)

#     from utils import extracting_subgraphs
#     all_graphs = extracting_subgraphs(net, all_indices, labels, h=1, u_features=u_features, v_features=v_features, max_node_label=3)
#     mydataset = MyDataset(all_graphs, root=f'data/{args.dataset}')

#     K = 5
#     skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
#     labels_np = np.array([d.y.item() for d in mydataset])

#     fold_metrics = []
#     all_train_losses = []
#     all_val_losses = []
#     all_fold_embeddings = []

#     for fold, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np)):
#         print('*' * 25, f'Fold {fold + 1}', '*' * 25)

#         from models import gGATLDA
#         in_features = mydataset[0].x.shape[1]
#         model = gGATLDA(in_features=in_features, num_gcn_layers=args.num_gcn_layers)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)

#         train_val_data = [mydataset[i] for i in train_val_index]
#         test_data = [mydataset[i] for i in test_index]

#         from sklearn.model_selection import StratifiedKFold
#         skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#         train_index, val_index = next(skf_inner.split(np.zeros(len(train_val_data)), [d.y.item() for d in train_val_data]))
#         train_data = [train_val_data[i] for i in train_index]
#         val_data = [train_val_data[i] for i in val_index]

#         fold_result = train_multiple_epochs(train_data, val_data, test_data, model, adj, fold + 1, max_epochs=50)
#         fold_metrics.append(fold_result)

#         train_loss, val_loss = zip(*fold_result['fold_losses'])
#         all_train_losses.append(train_loss)
#         all_val_losses.append(val_loss)

#         with torch.no_grad():
#             model.eval()
#             test_batch = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
#             for batch in test_batch:
#                 batch = batch.to(device)
#                 x = model.forward(batch)
#                 embedding = x.cpu().numpy()
#                 all_fold_embeddings.append((embedding, batch.y.cpu().numpy()))

#         print(f"Metrics for Fold {fold + 1}:")
#         for key, value in fold_result.items():
#             if key not in ["fold_losses", "epoch_accuracies", "fpr", "tpr", "precision_curve", "recall_curve", "truth", "predictions"]:
#                 print(f"{key}: {value:.4f}")

#     mean_train_losses = np.mean(all_train_losses, axis=0)
#     mean_val_losses = np.mean(all_val_losses, axis=0)

#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, len(mean_train_losses) + 1), mean_train_losses, label='Training Loss')
#     plt.plot(range(1, len(mean_val_losses) + 1), mean_val_losses, label='Validation Loss', color='orange')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss Over Epochs (Averaged Across Folds)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     metric_keys = [key for key in fold_metrics[0].keys() if key not in ["fold_losses", "epoch_accuracies", "fpr", "tpr", "precision_curve", "recall_curve", "truth", "predictions"]]
#     mean_metrics = { key: np.mean([fold_metric[key] for fold_metric in fold_metrics]) for key in metric_keys }
#     var_metrics = { key: np.var([fold_metric[key] for fold_metric in fold_metrics]) for key in metric_keys }

#     print("\nMean Metrics Across 5 Folds:")
#     for key, value in mean_metrics.items():
#         print(f"{key}: {value:.4f}")

#     print("\nVariance of Metrics Across 5 Folds:")
#     for key, value in var_metrics.items():
#         print(f"{key}: {value:.4f}")

#     # تحلیل آماری با داده واقعی
#     our_auc = np.array([0.9915, 0.9998, 0.9994, 1.0, 0.9927])
#     baseline_auc = np.array([0.9037, 0.9328, 0.9531, 0.9607, 0.9648])
#     t_stat, p_val = ttest_rel(our_auc, baseline_auc)
#     print(f"\nPaired t-test for AUC: p-value = {p_val:.4f}")
#     if p_val < 0.05:
#         print("Statistically significant difference.")
#     else:
#         print("No statistically significant difference.")

#     # t-SNE
#     print("\nRunning t-SNE on combined fold embeddings...")
#     combined_features = np.vstack([x[0] for x in all_fold_embeddings])
#     combined_labels = np.concatenate([x[1] for x in all_fold_embeddings])
#     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=seed)
#     tsne_result = tsne.fit_transform(combined_features)

#     plt.figure(figsize=(8, 6))
#     for label in np.unique(combined_labels):
#         indices = np.where(combined_labels == label)
#         plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=f"Class {int(label)}", alpha=0.6)
#     plt.title("t-SNE Visualization of Feature Embeddings")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     plot_roc_pr_curves(fold_metrics, k_folds=K)

# if __name__ == '__main__':
#     main()
