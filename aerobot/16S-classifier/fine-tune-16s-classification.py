import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from genslm import GenSLM, SequenceDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from torch.nn.functional import softmax
import pandas as pd
from sklearn.model_selection import train_test_split
import json 

# Assuming 'df' is your DataFrame
# df = pd.read_csv('your_data.csv')  # Load your data into a DataFrame


# Model and device setup
#model = GenSLM("genslm_25M_patric", model_cache_dir="/content/gdrive/MyDrive")
model_dir = "/groups/fischergroup/goldford/16s_embedding/models"
model = GenSLM("genslm_25M_patric", model_cache_dir=model_dir)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using {}".format(device))


# Load data
df = pd.read_csv('seqs.train.csv')  # Replace 'your_file.csv' with your actual CSV file path

# Splitting the DataFrame
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

sequences = train_df['sequence'].tolist()
sequences = [x.upper() for x in sequences]
labels = train_df['Oxygen_label'].tolist()

# Map labels to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
hidden_size = 512 

# Define the classifier
class TransformerClassifier(nn.Module):
    def __init__(self, base_model, num_classes,hidden_size):
        super(TransformerClassifier, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        emb = outputs.hidden_states[-1]
        emb = emb.mean(dim=1)
        return self.classifier(emb)

# Define the Dataset
class LabeledSequenceDataset(Dataset):
    def __init__(self, sequences, labels, seq_length, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            max_length=self.seq_length,
            padding="max_length",
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    
class LabeledSequenceDataset(SequenceDataset):
    def __init__(self, sequences, labels, *args, **kwargs):
        super().__init__(sequences, *args, **kwargs)
        self.labels = labels

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['label'] = self.labels[idx]
        return item



# Prepare DataLoader
dataset = LabeledSequenceDataset(sequences, labels, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset, batch_size=32)  # Adjust batch size as needed


# Load and evaluate the validation set
#val_df = pd.read_csv('seqs.test.csv')  # Replace 'your_val_file.csv' with your actual CSV file path

val_sequences = val_df['sequence'].tolist()
val_labels = le.transform(val_df['Oxygen_label'].tolist())  # Use the same LabelEncoder `le` to encode validation labels

val_dataset = LabeledSequenceDataset(val_sequences, val_labels, model.seq_length, model.tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=16)  # Adjust batch size as needed


# Instantiate and train the classifier
num_classes = 3  # Number of unique labels
clf_model = TransformerClassifier(model, num_classes,hidden_size)
clf_model.to(device)

criterion = nn.CrossEntropyLoss()

# Increase learning rate
learning_rate = 0.01
# Change optimizer
optimizer = optim.Adam(clf_model.parameters(), lr=learning_rate)


# Initialize lists to hold training summary
epochs_list = []
loss_list = []
train_accuracy_list = []
val_accuracy_list = []


num_epochs = 200

# Track the best validation accuracy
best_val_accuracy = 0.0
best_epoch = 0
best_model_results = {}  # Dictionary to hold the best model results


for epoch in range(num_epochs):  # Adjust the number of epochs as needed
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = clf_model(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Compute balanced accuracy on the training set after each epoch
    clf_model.eval()
    train_preds, train_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = clf_model(inputs, masks)
            pred_labels = softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            train_preds.extend(pred_labels.tolist())
            train_labels.extend(true_labels.tolist())

    train_acc = balanced_accuracy_score(train_labels, train_preds)

    # Compute balanced accuracy on the validation set after each epoch
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = clf_model(inputs, masks)
            pred_labels = softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            val_preds.extend(pred_labels.tolist())
            val_labels.extend(true_labels.tolist())

    val_acc = balanced_accuracy_score(val_labels, val_preds)
    
    # Append epoch loss and accuracies to the lists
    epochs_list.append(epoch+1)
    loss_list.append(loss.item())
    train_accuracy_list.append(train_acc)
    val_accuracy_list.append(val_acc)
    
    # Convert lists to DataFrame and save as CSV
    train_summary = pd.DataFrame({
        "Epoch": epochs_list,
        "Loss": loss_list,
        "Train_Accuracy": train_accuracy_list,
        "Val_Accuracy": val_accuracy_list
    })
    train_summary.to_csv('training_summary_tmp.csv', index=False)

    print(f'Epoch {epoch+1}, Loss: {loss.item()}, Train Accuracy: {train_acc*100:.2f}%, Val Accuracy: {val_acc*100:.2f}%')
    
    
    # Check if the current validation accuracy is the best one so far
    if val_acc > best_val_accuracy:
        
        best_val_accuracy = val_acc
        best_epoch = epoch + 1
        
        # Update the best model results
        best_model_results = {
            "best_epoch": best_epoch,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "loss": loss.item()
        }
        
        # Save the model as the best model
        best_model_path = f'models/clf_model_best.pt'
        torch.save(clf_model.state_dict(), best_model_path)
        print(f'New best model saved at epoch {best_epoch} with Val Accuracy: {best_val_accuracy*100:.2f}%')
        
        
    # Save model checkpoint
    #torch.save(clf_model.state_dict(), f'models/clf_model_{epoch+1}.pt')

    clf_model.train()  # set back to train mode for the next epoch

print('Finished Training')

# save best model results
# Save the best model results to a JSON file
with open('models/best_model_results.json', 'w') as json_file:
    json.dump(best_model_results, json_file, indent=4)


# Convert lists to DataFrame and save as CSV
train_summary = pd.DataFrame({
    "Epoch": epochs_list,
    "Loss": loss_list,
    "Train_Accuracy": train_accuracy_list,
    "Val_Accuracy": val_accuracy_list
})
train_summary.to_csv('training_summary.csv', index=False)