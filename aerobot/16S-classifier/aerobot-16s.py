import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from genslm import GenSLM, SequenceDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax
import pandas as pd
import pickle


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fasta_file', type=str, help='Path to the fasta file containing sequences to be classified.')
parser.add_argument('--weights_file', type=str, help='Path to the PyTorch model weights file.')
parser.add_argument('--output_file', type=str, help='Path to the output CSV file.')
parser.add_argument('--probabilities_file', type=str, help='Path to the output CSV file for class probabilities.', default=None)
args = parser.parse_args()



# Load the pickled LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


# Model and device setup
#model = GenSLM("genslm_25M_patric", model_cache_dir="/content/gdrive/MyDrive")
model_dir = "/groups/fischergroup/goldford/16s_embedding/models"
model = GenSLM("genslm_25M_patric", model_cache_dir=model_dir)


device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)
print("Using {}".format(device))


# Define the classifier
class TransformerClassifier(nn.Module):
    def __init__(self, base_model, num_classes,hidden_size):
        super(TransformerClassifier, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
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

    


# Prepare DataLoader
hidden_size = 512 

# Instantiate and train the classifier
num_classes = 3  # Number of unique labels
clf_model = TransformerClassifier(model, num_classes,hidden_size)
clf_model.to(device)



# Load the model weights
clf_model.load_state_dict(torch.load(args.weights_file, map_location=device))


# Load sequences from the fasta file
sequences = []
seq_headers = []
for seq_record in SeqIO.parse(args.fasta_file, "fasta"):
    sequences.append(str(seq_record.seq).upper())
    seq_headers.append(seq_record.id)

# Prepare DataLoader
dataset = SequenceDataset(sequences, model.seq_length, model.tokenizer)
dataloader = DataLoader(dataset, batch_size=32)  # Adjust batch size as needed


# Run the model on the data and gather predictions
clf_model.eval()
preds = []
probs = []
with torch.no_grad():
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        outputs = clf_model(inputs, masks)
        pred_labels = softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()
        pred_probs = softmax(outputs, dim=1).cpu().numpy()
        preds.extend(pred_labels.tolist())
        probs.extend(pred_probs.tolist())

# Decode labels back to original form
preds = le.inverse_transform(preds)

# Output to CSV
df = pd.DataFrame({
    'sequence_header': seq_headers,
    'predicted_label': preds,
})

df.to_csv(args.output_file, index=False)

# Output class probabilities to CSV
if args.probabilities_file:
    prob_df = pd.DataFrame(probs, columns=le.classes_)
    prob_df.insert(0, 'sequence_header', seq_headers)
    prob_df.to_csv(args.probabilities_file, index=False)
        
 