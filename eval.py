import os
import torch
from tqdm import tqdm
import pickle
from utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvalResults:
    def __init__(self, preds, labels, outputs):
        self.preds = preds
        self.labels = labels
        self.outputs = outputs
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    

def eval(model, val_loader, num_batches=16):
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []
    all_outputs = []

    # Iterate through the dataloader
    with torch.no_grad():
        for batchidx, (inputs, labels) in tqdm(enumerate(val_loader), total=num_batches):
            if batchidx >= num_batches:
                break

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            if isinstance(model, ViTForImageClassification):
                outputs = outputs.logits

            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  
            all_outputs.extend(outputs.cpu().numpy())
        
    return EvalResults(all_preds, all_labels, all_outputs)


def main():
    data_dir = "Data"

    # Create dataset DataFrame
    df = create_dataset_df(data_dir)

    print(df.info())

    BATCH_SIZE = 32

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data loader
    val_loader = create_val_loader(
        df,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        val_size=0.2
    )

    for model_name in os.listdir("models"):
        model_path = os.path.join("models", model_name)
        print(f"Loading model at {model_path}")
        model = load_model(model_path)
        print(f"{model_name} loaded successfully")

        print("Beginning evaluation")
        results = eval(model, val_loader, num_batches=16) # 16 batches of 32 images = 512 images being evaluated on
        save_path = rf"results\{model_name[:-4]}_eval_results.pkl"
        try:
            results.save(save_path)
            print(f"Results saved to {save_path}")
        except:
            print(f"Error saving results to {save_path}")
            continue



if __name__ == "__main__":
    
    main()



