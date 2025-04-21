

import scipy.io as scio
import pandas as pd
import numpy as np
import random
import os

def load_data(dataset):
    """
    بارگذاری داده‌ها برای دیتاست موردنظر.
    
    پارامتر:
        dataset: نام پوشه دیتاست (به عنوان مثال "mydata") که در مسیر raw_data/dataset قرار دارد.
        
    خروجی:
        u_features, v_features, net, labels, u_idx, v_idx, num_list
    """
    print("Loading lncRNAdisease dataset")
    # مسیر فایل MAT (تنها فایل موردنیاز تغییر می‌کند)
    path_dataset = os.path.join('raw_data', dataset, 'combined_datt.mat')
    data = scio.loadmat(path_dataset)
    
    # استخراج ماتریس ارتباط و ویژگی‌های مربوط به lncRNA و بیماری
    net = data['lncrna_disease_matrix']
    u_features = data['lncRNA_features']
    
    # بارگذاری فایل CSV برای شباهت بیماری‌ها
    disSim_path = os.path.join('raw_data', dataset, 'disease_similarity_matrix.csv')
    disSim_data = pd.read_csv(disSim_path, header=None)
    v_features = np.array(disSim_data)
    
    print("Shape of net.T:", net.T.shape)
    print("Shape of v_features:", v_features.shape)
    
    # در صورت عدم تطابق ابعاد، پد کردن جهت یکپارچه‌سازی
    if net.T.shape[0] != v_features.shape[0]:
        print("Mismatch in number of diseases between net.T and v_features.")
        max_rows = max(net.T.shape[0], v_features.shape[0])
        net_T_padded = np.zeros((max_rows, net.shape[1]))
        net_T_padded[:net.T.shape[0], :] = net.T
        v_features_padded = np.zeros((max_rows, v_features.shape[1]))
        v_features_padded[:v_features.shape[0], :] = v_features
        print("After padding:")
        print("Shape of net_T_padded:", net_T_padded.shape)
        print("Shape of v_features_padded:", v_features_padded.shape)
        v_features = np.hstack((net_T_padded, v_features_padded))
    else:
        v_features = np.hstack((net.T, v_features))
    
    num_list = [len(u_features), len(v_features)]
    # الحاق ماتریس شبکه به ویژگی‌های lncRNA
    u_features = np.hstack((u_features, net))
    
    # تنظیم ابعاد با افزودن یک ردیف صفر (برای تطبیق ایندکس‌ها)
    a = np.zeros((1, u_features.shape[1]), int)
    u_features = np.vstack((a, u_features))
    
    # تنظیم ابعاد ویژگی‌های بیماری با افزودن یک ردیف صفر
    b = np.zeros((1, v_features.shape[1]), int)
    v_features = np.vstack((b, v_features))
    
    # نمونه‌گیری برای نمونه‌های مثبت و منفی
    num_lncRNAs = net.shape[0]
    num_diseases = net.shape[1]
    row, col = net.nonzero()
    perm = random.sample(range(len(row)), len(row))
    row, col = row[perm], col[perm]
    sample_pos = (row, col)
    print("The number of all positive samples:", len(sample_pos[0]))
    
    print("Sampling negative links for train and test")
    X = np.ones((num_lncRNAs, num_diseases))
    net_neg = X - net
    row_neg, col_neg = net_neg.nonzero()
    perm_neg = random.sample(range(len(row_neg)), len(row))
    row_neg, col_neg = row_neg[perm_neg], col_neg[perm_neg]
    sample_neg = (row_neg, col_neg)
    print("The number of all negative samples:", len(sample_neg[0]))
    
    u_idx = np.hstack([sample_pos[0], sample_neg[0]])
    v_idx = np.hstack([sample_pos[1], sample_neg[1]])
    labels = np.hstack([[1]*len(sample_pos[0]), [0]*len(sample_neg[0])])
    
    # افزودن ردیف و ستون صفر به ماتریس شبکه جهت تطبیق ایندکس‌ها
    l1 = np.zeros((1, net.shape[1]), int)
    net = np.vstack([l1, net])
    l2 = np.zeros((net.shape[0], 1), int)
    net = np.hstack([l2, net])
    
    u_idx = u_idx + 1
    v_idx = v_idx + 1
    
    return u_features, v_features, net, labels, u_idx, v_idx, num_list

