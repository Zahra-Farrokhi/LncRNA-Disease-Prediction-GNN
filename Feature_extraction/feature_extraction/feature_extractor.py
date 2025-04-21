# feature_extraction/feature_extractor.py

import torch
import numpy as np
import pandas as pd
import logging

def extract_features(model, dataloader, device, output_csv="lstm_features.csv"):
    model.eval()
    features_list = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            _, features = model(batch_x)
            features_list.append(features.cpu().numpy())
    all_features = np.vstack(features_list)
    feature_columns = [f'Feature_{i+1}' for i in range(all_features.shape[1])]
    df = pd.DataFrame(all_features, columns=feature_columns)
    try:
        df.to_csv(output_csv, index=False)
        logging.info(f"Features successfully saved to {output_csv}")
    except Exception as e:
        logging.error(f"Failed to save features to CSV: {e}")
