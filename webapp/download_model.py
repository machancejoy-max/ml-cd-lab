import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

def main():
    print("Downloading tokenizer and model...")

    # Choose a sentiment model that includes a classification head
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    # Create output directory
    os.makedirs("webapp", exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained("webapp/tokenizer")
    print("Tokenizer saved to webapp/tokenizer")

    # Save full model (PyTorch object)
    model_path = "webapp/model_full.pt"
    torch.save(model, model_path)
    print("Model saved to", model_path)

if __name__ == "__main__":
    main()
