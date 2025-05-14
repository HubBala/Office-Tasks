# Task 22 -- Reinforcement Learning model using the kmeans and q_table 

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Diabetes dataset.csv")
# columns with invalid Zeroes
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

# Cluster into states using KMeans (you can adjust n_clusters)
kmeans = KMeans(n_clusters=100, random_state=42)
states = kmeans.fit_predict(features_scaled)

# Save the scaler and kmeans for later use in Streamlit_Task22.py
with open("scalerRec.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Define treatment actions
actions = ["Diet", "Medication", "Exercise"]
n_actions = len(actions)

# Q-table: [state x actions]
q_table = np.zeros((100, n_actions))

# RL hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.1
episodes = 500

# Reward function based on action success
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
        state = states[i]  # Using KMeans clusters as states
        
        # Exploration or Exploitation
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)  # Random action (exploration)
        else:
            action = np.argmax(q_table[state])  # Best action (exploitation)
        
        # Calculate reward based on action and label
        reward = calculate_reward(labels[i], action)
        next_state = state  # For one-step episodes, next state is the same as current state
        
        # Update Q-table using Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        total_reward += reward

    # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Optionally, print the total reward of each episode for debugging
    if episode % 50 == 0:  # Print every 50 episodes
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Save the Q-learning model (Q-table)
with open("q_table_Rec.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete.")
