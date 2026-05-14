import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------
# Load full dataset
# -----------------------

df = pd.read_csv("data_clean.csv")

# -----------------------
# Split
# -----------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# -----------------------
# Save permanently
# -----------------------

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train/Test split saved.")