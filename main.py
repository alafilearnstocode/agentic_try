import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from sklearn.preprocessing import StandardScaler

CAN = {0: "HOLD", 1: "BUY", 2: "SELL"}

def load_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    scaler = StandardScaler()
    data[['Close', 'SMA_5', 'SMA_20', 'Return']] = scaler.fit_transform(data[['Close', 'SMA_5', 'SMA_20', 'Return']])
    return data

def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5'].iloc[0]),
        float(data.loc[index, 'SMA_20'].iloc[0]),
        float(data.loc[index, 'Return'].iloc[0])
    ])

class Environment: # the agent that interacts with the environment
    def __init__(self, data):
        self.data = data
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.index = 0

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.index = 0
        return get_state(self.data, self.index)
    
    def step(self, action):
        price = float(self.data.loc[self.index, 'Close'])
        reward = 0

        if action == 1 and self.balance >= price:
            shares_bought = max(1, int((0.1 * self.balance) // price))  # buy up to 10% of balance
            self.shares_held += shares_bought
            self.balance -= shares_bought * price
        elif action == 2 and self.shares_held > 0:
            shares_sold = max(1, int(0.1 * self.shares_held))  # sell 10% of holdings
            self.balance += shares_sold * price
            self.shares_held -= shares_sold
        
        if self.index > 0:
            prev_price = float(self.data.loc[self.index - 1, 'Close'])
            prev_value = self.balance + self.shares_held * prev_price
            curr_value = self.balance + self.shares_held * price
            reward = curr_value - prev_value
        
        self.index += 1
        done = self.index >= len(self.data) - 1

        if done:
            total_value = self.balance + self.shares_held * price # revisit
            reward = total_value - (self.balance + self.shares_held * prev_price)

        next_state = get_state(self.data, self.index) if not done else None
        return next_state, reward, done, {}
    
class DQN(nn.Module): # the decision making neural network
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
            
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # valuing our future rewards
        self.epsilon = 1.0 # random action
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # prediceted and the target value
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(CAN.keys()))
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else None
            target = reward
            if not done and next_state is not None:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()

def train_agent(data, episodes, batch_size):
    env = Environment(data)
    state_size = 4
    action_size = len(CAN)
    agent = DQNAgent(state_size, action_size)
    total_rewards = []
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.replay(batch_size)
        total_rewards.append(total_reward)
    return agent, total_rewards

def test_agent(agent, data):
    test_env = Environment(data)
    state = test_env.reset()
    done = False
    records = []
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = test_env.step(action)
        records.append({
            'Action': CAN[action],
            'Balance': test_env.balance,
            'Shares Held': test_env.shares_held
        })
        state = next_state if next_state is not None else state
    final_balance = test_env.balance + test_env.shares_held * float(data.loc[test_env.index - 1, 'Close'].iloc[0])
    profit = final_balance - test_env.initial_balance
    results_df = pd.DataFrame(records)
    return final_balance, profit, results_df

st.title("Reinforcement Learning Stock Trader 📈")

symbol = st.sidebar.text_input("Stock Symbol", value="NVDA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-07-31"))
episodes = st.sidebar.slider("Episodes", min_value=10, max_value=1000, value=500, step=10)
batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)

if st.sidebar.button("Run Training"):
    with st.spinner("Downloading data and training the agent..."):
        data = load_data(symbol, start_date, end_date)
        agent, total_rewards = train_agent(data, episodes, batch_size)
    st.success("Training complete!")

    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Rewards Over Episodes")
    st.pyplot(fig)

    final_balance, profit, results_df = test_agent(agent, data)
    st.subheader("Test Run Results")
    st.line_chart(results_df[['Balance', 'Shares Held']])
    st.dataframe(results_df)
    st.metric("Final Balance", f"${final_balance:.2f}")
    st.metric("Profit", f"${profit:.2f}")
