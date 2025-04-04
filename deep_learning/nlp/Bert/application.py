import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
import datasets


def main(device):
    dataset = datasets.load_dataset("imdb")
    train_iter, test_iter = dataset["train"], dataset["test"]

    # Check how many category in this dataset
    labels = [example["label"] for example in train_iter]
    num_labels = len(np.unique(labels))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels).to(device)


    def collate_batch(batch):
        label_list, text_list, attention_list = [], [], []

        for example in batch:
            label_list.append(example["label"])
            text_list.append(example["text"])

        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        label_list = torch.tensor(label_list)

        return inputs["input_ids"].to(device), inputs["attention_mask"].to(device), label_list.to(device)


    train_dl = DataLoader(
        train_iter,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_batch    # custom defined processed function
    )

    test_dl = DataLoader(
        test_iter,
        batch_size=16,
        collate_fn=collate_batch    # custom defined processed function
    )


    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)


    def train(dataloader):
        total_acc, total_count, total_loss = 0, 0, 0
        model.train()

        for text, mask, label in dataloader:
            optimizer.zero_grad()
            outputs = model(text, token_type_ids=None, attention_mask=mask, labels=label)
            loss = outputs.loss
            loss.backward()
            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


    def test(dataloader):
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for text, mask, label in dataloader:
                outputs = model(text, token_type_ids=None, attention_mask=mask, labels=label)
                logits = outputs.logits
                loss = outputs.loss
                eval_loss += loss.item()
                nb_eval_steps += 1

                logits = logits.detach().cpu().numpy()
                label_ids = label.to("cpu").numpy()
                predictions.extend(logits.argmax(axis=1).flatten())
                true_labels.extend(label_ids.flatten())
        
        eval_loss = eval_loss / nb_eval_steps
        accuracy = accuracy_score(true_labels, predictions)
        _, _, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="macro")

        print(f"Loss: {nb_eval_examples:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    epochs = 15
    
    for epoch in range(epochs):
        print("epoch:", epoch)
        train(train_dl)
        test(test_dl)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
