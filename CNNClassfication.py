import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io

# pip install pyarrow huggingface_hub
# https://huggingface.co/datasets/dpdl-benchmark/places365-mini-sample-hard

df_train = pd.read_parquet("./pytorch_src/project/data/places365-mini-sample-hard-train.parquet")
df_test = pd.read_parquet("./pytorch_src/project/data/places365-mini-sample-hard-test.parquet")

# print(f'Preview of training data:\n{df_train.head()}')
# print(f'Shape of training and test data: {df_train.shape}, {df_test.shape}')

# print(f'Label distribution in training data:\n{df_train["label"].value_counts()}')
# print(f'Label distribution in test data:\n{df_test["label"].value_counts()}')

# print(df_train['image'])


def show_first_images(df, i=10):
    fig, axes = plt.subplots(6, 4, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.head(i).iterrows()):
        image_bytes = row["image"]["bytes"]
        label = row["label"]

        image = Image.open(io.BytesIO(image_bytes))
        
        ax.imshow(image)
        ax.set_title(f"{label} {image.size}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_first_images(df_train)
