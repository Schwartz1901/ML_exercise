import numpy as np

def entropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return - np.sum(probs * np.log2(probs))

train_data = [
    {"ID": 1, "Age": 35, "CreditScore": 720, "Education": 16, "RiskLevel": "Low"},
    {"ID": 2, "Age": 28, "CreditScore": 650, "Education": 14, "RiskLevel": "High"},
    {"ID": 3, "Age": 45, "CreditScore": 750, "Education": "missing", "RiskLevel": "Low"},
    {"ID": 4, "Age": 31, "CreditScore": 600, "Education": 12, "RiskLevel": "High"},
    {"ID": 5, "Age": 52, "CreditScore": 780, "Education": 18, "RiskLevel": "Low"},
    {"ID": 6, "Age": 29, "CreditScore": 630, "Education": 14, "RiskLevel": "High"},
    {"ID": 7, "Age": 42, "CreditScore": 710, "Education": 16, "RiskLevel": "Low"},
    {"ID": 8, "Age": 33, "CreditScore": 640, "Education": 12, "RiskLevel": "High"},
]

### Question 1
risk_level =np.array([d["RiskLevel"] for d in train_data])

init_entropy = entropy(risk_level)

left_subset = [d for d in train_data if d["CreditScore"] <= 650]
right_subset = [d for d in train_data if d["CreditScore"] > 650]

left_entropy = entropy(d["RiskLevel"] for d in left_subset)
right_entropy = entropy(d["RiskLevel"] for d in right_subset)

# Compute weighted entropy
total_size = len(train_data)

left_weight = len(left_subset) / total_size
right_weight = len(right_subset) / total_size

weighted_entropy = left_weight * left_entropy + right_weight * right_entropy

information_gain = init_entropy - weighted_entropy

print(f"Initial Entropy: {init_entropy:.4f}")
print(f"Left Entropy: {left_entropy:.4f}, Right Entropy: {right_entropy:.4f}")
print(f"Weighted Entropy: {weighted_entropy:.4f}")
print(f"Information Gain for CreditScore=650: {information_gain:.4f}")

