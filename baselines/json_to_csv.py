import pandas as pd
import json

# Load your JSON data
with open("readouts_all_methods_datasets_v1.json", "r") as f:  #
    data = json.load(f)

columns = [
    "Metrics", 
    "CFP R Acc", "CFP F Acc", "CFP MIA",
    "MRI R Acc", "MRI F Acc", "MRI MIA",
    "CIFAR-10 R Acc", "CIFAR-10 F Acc", "CIFAR-10 MIA",
    "fashionMNIST R Acc", "fashionMNIST F Acc", "fashionMNIST MIA",
    "medMNIST R Acc", "medMNIST F Acc", "medMNIST MIA"
]
df = pd.DataFrame(columns=columns)

datasets = ["CFP", "MRI", "cifar10", "fashionmnist", "medmnist"]
metrics = ["Retrain", "Finetune", "NegGrad", "CFK", "EUK", "SCRUB", "Chen Et. Al.", "RAVI"]

for metric in metrics:
    row = {"Metrics": metric}
    for dataset in datasets:
        if metric.lower() in data:
            dataset_data = data[metric.lower()].get(dataset.lower(), {})
            r_acc = dataset_data.get("retain_error", None)
            f_acc = dataset_data.get("forget_error", None)
            mia_mean = dataset_data.get("MIA_mean", None)
            row[f"{dataset} R Acc"] = r_acc
            row[f"{dataset} F Acc"] = f_acc
            row[f"{dataset} MIA"] = mia_mean
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

df.to_csv("output_table.csv", index=False)
