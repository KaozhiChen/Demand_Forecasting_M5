import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "training_data.csv")

# Load data from training_data.csv
df = pd.read_csv(data_path, parse_dates=["date"])

# Select a typical item 
item_id = 'FOODS_1_001'
subset = df[df['item_id'] == item_id]

# Check if item exists
if len(subset) == 0:
    print(f"Item {item_id} not found. Available items:")
    print(df['item_id'].unique()[:10])
    # Use the first available item
    item_id = df['item_id'].iloc[0]
    subset = df[df['item_id'] == item_id]
    print(f"Using item: {item_id}")

# Set plotting style
try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')

# --- Sales Distribution Histogram (showing sparsity) ---
plt.figure(figsize=(8, 5))
# Calculate bins, one bin per integer
max_sales = int(subset['sales'].max())
bins = np.arange(0, max_sales + 2) - 0.5 

sns.histplot(subset['sales'], bins=bins, kde=False, color='#4c72b0', edgecolor='black')

plt.title(f'Sales Distribution: {item_id}\n(High Frequency of Zero Sales)', fontsize=14, fontweight='bold')
plt.xlabel('Daily Sales Volume', fontsize=12)
plt.ylabel('Frequency (Days)', fontsize=12)
plt.xticks(range(0, min(max_sales + 1, 50)))  # Limit x-axis ticks to avoid overcrowding
plt.grid(axis='y', alpha=0.3)  # Show y-axis grid for better readability

plt.tight_layout()
plt.savefig('sales_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
