import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def pad_sequences(data, pad_value=0):
    """Pad sequences in the dataset to ensure they have the same length."""
    max_len = max(len(item) for item in data)
    padded_data = [item + [pad_value] * (max_len - len(item)) for item in data]
    return padded_data

# Load the data
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Assuming data_dict['data'] is a list of lists with varying lengths
# Pad data so all sequences have the same length
padded_data = pad_sequences(data_dict['data'])

# Convert to NumPy array
data = np.array(padded_data)
labels = np.array(data_dict['labels'])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)
model.fit(x_train, y_train)

# Make predictions and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
