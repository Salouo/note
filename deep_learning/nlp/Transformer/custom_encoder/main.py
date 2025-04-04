import torch
import datasets
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.utils.data import DataLoader
import math
from torch.optim import lr_scheduler
from custom_multi_head_attention import TransformerEncoder


def main(device, train_data, test_data, vocab_size):
    # If tokenizer has already been trained, load it. If not, a new tokenizer will be trained.
    try:
        tokenizer = Tokenizer.from_file("./my_tokenizer.json")
        print("Loaded existed Tokenizer...")

    except Exception as e:
        print("Did not find Tokenzier JSON, training again!")
        # Create BPE Tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size = vocab_size,
            special_tokens = ["[PAD]", "[UNK]"]
        )

        # Define a generator for getting a tokenizer
        def corpus_generator():
            for example in train_data:
                # Return a text at each iteration
                yield example["text"]

        # Train a tokenizer
        tokenizer.train_from_iterator(corpus_generator(), trainer=trainer)

        # Save the trained tokenizer
        tokenizer.save("./my_tokenizer.json")


    def collate_batch(batch):
    ###########################################################################################
    # This function is used to process the texts in each batch, return the processed results. #
    ###########################################################################################
        label_list, text_list = [], []
        for example in batch:
            # Append label into label_list
            label_list.append(example["label"])

            # Convert the text to their corresponding ids, and append them into text_list
            process_text = torch.tensor(tokenizer.encode(example["text"]).ids, dtype=torch.int64)
            text_list.append(process_text)
        
        # Convert label_list to a tensor type
        label_list = torch.tensor(label_list)

        # padding
        text_list = torch.nn.utils.rnn.pad_sequence(text_list)

        # Return the processed results
        return label_list.to(device), text_list.to(device)
        

    train_dl = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_batch    # custom defined processed function
    )

    test_dl = DataLoader(
        test_data,
        batch_size=128,
        collate_fn=collate_batch    # custom defined processed function
    )


    ###########################################################################################
    #                               Positional Encoder                                        #
    ###########################################################################################
    class PositionalEncoder(nn.Module):
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()
            self.max_len = max_len
            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)   # Add a dimension; shape: (length, 1)
            # Use logarithmic operations to convert exponentiation into multiplication
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

            pe = torch.zeros(max_len, 1, d_model)   # 1 is for batch_size
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)  # Register pe to a model parameter, but this parameter cannot be updated
        
        def forward(self, x):
            x = x[:self.max_len] + self.pe[:x.size(0)]
            return self.dropout(x)


    ###########################################################################################
    #                               Custom Transformer Model                                  #
    ###########################################################################################
    class TransformerModel(nn.Module):

        def __init__(self, ntokens, d_model, nhead, num_layers, dropout, d_hid, out_class):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings=ntokens, embedding_dim=d_model)     # embedding layer
            self.pos_encoder = PositionalEncoder(d_model=d_model, dropout=dropout)   # positional encoder
            self.transformer_encoder = TransformerEncoder(num_layers, d_model, nhead, d_hid)
            self.fc = nn.Linear(in_features=d_model, out_features=out_class)     # For binary classification task

        def forward(self, x):
            x = self.embedding(x)   # embedding
            x = self.pos_encoder(x)     # position encoding
            x = x.permute(1, 0, 2)
            x = self.transformer_encoder(x)     # attention, shape: (length, batch_size, d_model)
            x = x.permute(1, 0, 2)
            x = x[0, :, :]  # We can use the feature of the first word to represent the fature of the whole text. Important!!
            x = self.fc(x)
            return x

    ntokens = vocab_size
    d_model = 200
    d_hid = 2048    # size of hidden layer of FFN
    num_layers = 2
    nhead = 2
    dropout = 0.2
    out_class = 2

    model = TransformerModel(ntokens, d_model, nhead, num_layers, dropout, d_hid, out_class).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


    def train(dataloader):
        total_acc, total_count, total_loss = 0, 0, 0
        model.train()

        for label, text in dataloader:
            pred = model(text)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_acc += (pred.argmax(1) == label).sum().item()
                total_count += label.size(0)
                total_loss += loss.item() * label.size(0)

        return total_loss / total_count, total_acc / total_count

    
    def test(dataloader):
        model.eval()
        total_acc, total_count, total_loss = 0, 0, 0

        with torch.no_grad():
            for label, text in dataloader:
                pred = model(text)
                loss = loss_fn(pred, label)
                total_acc += (pred.argmax(1) == label).sum().item()
                total_count += label.size(0)
                total_loss += loss.item() * label.size(0)
        
        return total_loss / total_count, total_acc / total_count

    
    def fit(epochs, train_dl, test_dl):
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        for epoch in range(epochs):
            epoch_loss, epoch_acc = train(train_dl)
            epoch_test_loss, epoch_test_acc = test(test_dl)
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)
            exp_lr_scheduler.step()

            template = ("epoch: {:2d}, train_loss: {:.5f}, train_acc: {:.1f}%,"
            "test_loss: {:.5f}, test_acc: {:.1f}%")
            print(template.format(epoch, epoch_loss, epoch_acc * 100, epoch_test_loss, epoch_test_acc * 100))
        
        print("Done!")

        return train_loss, test_loss, train_acc, test_acc


    epochs = 50
    train_loss, test_loss, train_acc, test_acc = fit(epochs, train_dl, test_dl)


if __name__ == "__main__":
    # Check if we have GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtain data (0: 'neg', 1: 'pos')
    dataset = datasets.load_dataset("imdb")
    train_data, test_data = dataset["train"], dataset["test"]

    # Define the size of vocab
    vocab_size = 40000

    main(device, train_data, test_data, vocab_size)
