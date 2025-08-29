# AI Stock Trading Agent (Reinforcement Learning)

This project is a Streamlit app showcasing an AI trading agent built with reinforcement learning (Deep Q-Networks). It allows users to train and test a reinforcement learning agent to trade stocks based on historical data.

## Features

- Choose stock symbol to trade
- Set training parameters for the reinforcement learning agent
- View training rewards over time
- Test the trained model and view results including balances and profit

## How It Works

The agent uses a Deep Q-Network to learn buy/hold/sell strategies from historical stock data, with experience replay and epsilon-greedy exploration.

## Installation

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app with:

```
streamlit run main.py
```

## Deployment

To deploy this app on Streamlit Cloud, make sure to include the `requirements.txt` file with the project to install all dependencies automatically.