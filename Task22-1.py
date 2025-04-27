# reinforcement learning using Q-Table

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.impute import SimpleImputer
# Load dataset
df = pd.read_csv("Diabetes dataset.csv")

cols_with_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)

# 0s with mean vbalues 
imputer = SimpleImputer(strategy="mean")
df[cols_with_zero_invalid] = imputer.fit_transform(df[cols_with_zero_invalid])


features = df.drop(columns=["Outcome"]).values
labels = df["Outcome"].values

# Normalize data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save the scaler for later use in inference
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define treatment actions
actions = ["Diet", "Medication", "Exercise"]
n_actions = len(actions)

# Q-table: Dictionary with state (tuple) as key and array of action-values as value
q_table = {}

# RL hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.1
episodes = 500

# Reward rule based on action success
def calculate_reward(label, action):
    success_probs = [0.4, 0.6, 0.5]  # diet, medication, exercise success probabilities
    success = np.random.rand() < success_probs[action]
    if success and label == 1:  # Correct treatment for positive label
        return 100
    elif success:  # Correct treatment for negative label
        return 10
    else:  # Incorrect treatment
        return -10

# Training loop
for episode in range(episodes):
    total_reward = 0
    for i in range(len(features_scaled)):
        # Use scaled and rounded feature tuple as the state
        state = tuple(features_scaled[i].round(1))

        # Initialize Q-values for unseen states
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)

        # Exploration or Exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)  # Random action
        else:
            action = np.argmax(q_table[state])  # Best action

        # Calculate reward
        reward = calculate_reward(labels[i], action)
        next_state = state  # Episode ends here, so no transition

        # Q-learning update
        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * np.max(q_table[state]) - q_table[state][action]
        )

        total_reward += reward

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Save the Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete.")

# Show best actions for first 10 patients
print("\nRecommended actions for first 10 patients:")
for i in range(10):
    state = tuple(features_scaled[i].round(1))
    best_action = np.argmax(q_table[state])
    print(f"Patient {i}: Recommend -> {actions[best_action]}")
