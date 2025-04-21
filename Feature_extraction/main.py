# main.py

import torch
from data.data_loader import preprocess_data, LNCRNADataset
from models.lstm_model import LSTMModel
from train.trainer import train_epoch, validate
from feature_extraction.feature_extractor import extract_features
from utils.logger import setup_logging
from utils.early_stopping import EarlyStopping
import tracemalloc
import logging
import sys
import os

def main():
    try:
        tracemalloc.start()
        setup_logging()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        input_size = 4
        hidden_size = 512
        num_layers = 2
        num_classes = 4
        sequence_length = 8000
        learning_rate = 0.001
        epochs = 50
        batch_size = 4
        step_size = 10
        bidirectional = False
        dropout = 0.5
        patience = 10
        fasta_file = "lncrna_sequences.fasta"
        feature_csv = "lstm_features.csv"
        model_checkpoint = "best_model.pth"
        if not os.path.exists(fasta_file):
            logging.error(f"FASTA file '{fasta_file}' not found.")
            sys.exit(1)
        encoded_sequences = preprocess_data(fasta_file, sequence_length)
        dataset = LNCRNADataset(encoded_sequences, sequence_length, step=step_size)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                                 num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                               num_workers=2, pin_memory=True)
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes, 
                         bidirectional=bidirectional, dropout=dropout).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001, path=model_checkpoint)
        logging.info("Starting training loop...")
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = validate(model, val_loader, criterion, device)
            logging.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy*100:.2f}% - Val Precision: {val_precision*100:.2f}% - Val Recall: {val_recall*100:.2f}% - Val F1-Score: {val_f1*100:.2f}%")
            early_stopping(val_loss, model, logging)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered. Stopping training.")
                break
            if epoch % 10 == 0:
                checkpoint_path = f"model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved at epoch {epoch}.")
        logging.info("Training completed.")
        if os.path.exists(model_checkpoint):
            logging.info(f"Loading the best model from {model_checkpoint}...")
            model.load_state_dict(torch.load(model_checkpoint))
        else:
            logging.warning(f"Best model checkpoint {model_checkpoint} not found. Using the last trained model.")
        logging.info("Starting feature extraction...")
        extract_features(model, val_loader, device, output_csv=feature_csv)
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Current memory usage: {current / 10**6:.2f} MB; Peak was {peak / 10**6:.2f} MB")
        tracemalloc.stop()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
