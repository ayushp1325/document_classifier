from transformers import BertTokenizer, BertModel
import torch

def get_bert_embeddings(texts):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode the texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Get the embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The embeddings are in the last hidden state
    embeddings = outputs.last_hidden_state
    return embeddings

# Test the function
texts = ['This is a test.', 'This is another test.']
embeddings = get_bert_embeddings(texts)
print(embeddings)

