import pandas as pd

# pip install pyarrow huggingface_hub
# Hugging Face: https://huggingface.co/datasets/dpdl-benchmark/places365-mini-sample-hard
# Kaggle: https://www.kaggle.com/datasets/benjaminkz/places365/data
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}

df_train = pd.read_parquet("hf://datasets/dpdl-benchmark/places365-mini-sample-hard/" + splits["train"])
df_train.to_parquet("./data/places365-mini-sample-hard-train.parquet")

df_test = pd.read_parquet("hf://datasets/dpdl-benchmark/places365-mini-sample-hard/" + splits["test"])
df_test.to_parquet("./data/places365-mini-sample-hard-test.parquet")

print(f'Preview of training data:\n{df_train.head()}')
print(f'Shape of training and test data: {df_train.shape}, {df_test.shape}')

print(f'Label distribution in training data:\n{df_train["label"].value_counts()}')
print(f'Label distribution in test data:\n{df_test["label"].value_counts()}')
