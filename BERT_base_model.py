
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing and Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example labeled data
texts = ["I hate this movie, it's terrible.", "This is a great day!"]
labels = [1, 0]  # 1 for hate speech, 0 for non-hate speech

# Tokenize and encode the texts
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = tokenized_texts["input_ids"].to(device)
attention_mask = tokenized_texts["attention_mask"].to(device)
labels = torch.tensor(labels).to(device)

# Define BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 5
total_steps = len(input_ids) * num_epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Data loader
batch_size = 2
dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids_batch, attention_mask_batch, labels_batch = batch
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
        loss = outputs.loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Adjust learning rate
    scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss}")

# Save the trained model
model.save_pretrained("bert_hate_speech_classification")
tokenizer.save_pretrained("bert_hate_speech_classification")


# Example test data
test_texts = ["This movie is awful.", "I love this book!"]

# Tokenize and encode the test texts
tokenized_test_texts = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
input_ids_test = tokenized_test_texts["input_ids"].to(device)
attention_mask_test = tokenized_test_texts["attention_mask"].to(device)

# Testing/Prediction
model.eval()

with torch.no_grad():
    outputs = model(input_ids_test, attention_mask=attention_mask_test)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).tolist()

label_map = {0: "Non-Hate Speech", 1: "Hate Speech"}

for text, label in zip(test_texts, predicted_labels):
    print(f"Text: {text}")
    print(f"Predicted Label: {label_map[label]}")
    print()


