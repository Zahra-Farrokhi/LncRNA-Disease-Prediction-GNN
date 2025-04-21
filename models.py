import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_add_pool

class gGATLDA(torch.nn.Module):
    def __init__(self, in_features, gconv=GATConv, latent_dim=[64, 64, 64], num_gcn_layers=2, side_features=False, n_side_features=0):
        """
        پارامترها:
            in_features: ابعاد ورودی (تعداد ویژگی‌های اولیه هر گره)
            gconv: کلاس لایه گراف کانولوشنال توجهی (پیش‌فرض GATConv)
            latent_dim: لیستی از ابعاد نهفته برای لایه‌های GAT
            num_gcn_layers: تعداد لایه‌های GCN
            side_features: اگر ویژگی‌های جانبی نیز وجود داشته باشد (اختیاری)
            n_side_features: تعداد ویژگی‌های جانبی
        """
        super(gGATLDA, self).__init__()
        self.side_features = side_features
        self.n_side_features = n_side_features

        # تعریف لایه‌های GCN به‌صورت داینامیک
        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_features, 64))
        self.gcn_bns.append(nn.BatchNorm1d(64))
        for _ in range(1, num_gcn_layers):
            self.gcn_layers.append(GCNConv(64, 64))
            self.gcn_bns.append(nn.BatchNorm1d(64))

        # تعریف لایه‌های GAT
        self.gat1 = gconv(64, latent_dim[0], heads=8, dropout=0.2)
        self.gat2 = gconv(latent_dim[0]*8, latent_dim[1], heads=6, dropout=0.5)
        self.gat3 = gconv(latent_dim[1]*6, latent_dim[2], heads=4, dropout=0.5)

        # نرمال‌سازی برای لایه‌های GAT
        self.bn1 = nn.BatchNorm1d(latent_dim[0]*8)
        self.bn2 = nn.BatchNorm1d(latent_dim[1]*6)

        # لایه‌های کاملاً متصل برای دسته‌بندی
        self.fc1 = nn.Linear(latent_dim[2]*4, 128)
        self.fc2 = nn.Linear(128, 2)

        # Dropout جهت کاهش overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, data):
        """
        پارامتر:
            data: شی PyG Data شامل (x, edge_index, batch)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # عبور از لایه‌های GCN با اعمال ReLU، BatchNorm و Dropout
        for layer, bn in zip(self.gcn_layers, self.gcn_bns):
            x = F.relu(layer(x, edge_index))
            x = bn(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # عبور از لایه‌های GAT با فعال‌سازی ELU و BatchNorm
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        x = F.elu(x)

        # global pooling جهت ترکیب نمایه‌های تمامی گره‌ها
        x = global_add_pool(x, batch)
        
        # لایه‌های کاملاً متصل
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # خروجی خام (logits)
        return x

    def get_embedding(self, data):
        """
        مشابه forward اما تا قبل از لایه نهایی fc2 اجرا می‌شود.
        این متد بردار ویژگی (embedding) استخراج‌شده از گراف را برمی‌گرداند.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # عبور از لایه‌های GCN
        for layer, bn in zip(self.gcn_layers, self.gcn_bns):
            x = F.relu(layer(x, edge_index))
            x = bn(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # عبور از لایه‌های GAT
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        x = F.elu(x)
        
        # global pooling
        x = global_add_pool(x, batch)
        
        # لایه fc1 به عنوان خروجی embedding
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x


