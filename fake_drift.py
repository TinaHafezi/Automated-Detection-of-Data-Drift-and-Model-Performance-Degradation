import pandas as pd
import numpy as np

df = pd.read_csv("Train model/current.csv")

# Add strong shift
df["MonthlyCharges"] = df["MonthlyCharges"] + 100  

# df["MonthlyCharges"] = df["MonthlyCharges"] * 10  # a different kind of drift

df.to_csv("Train model/current.csv", index=False)



