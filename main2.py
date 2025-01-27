#chaturya
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data
df = pd.read_csv('soapnutshistory.csv')

# Handle missing values
df['Product Price'] = np.random.uniform(low=50, high=150, size=df.shape[0])  # Replace missing prices
imputer = SimpleImputer(strategy='mean')
X = df.iloc[:, 2:7].values  # Assuming columns 3-7 are features
y = df['Product Price'].values

# Impute missing values
X_imputed = imputer.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train a sales prediction model
sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
sales_model.fit(X_train, y_train)

# Step 2: Define the Q-Learning Environment
class PricingEnv:
    def __init__(self, model, price_range=(50, 150), price_step=5):
        self.model = model
        self.price_range = price_range
        self.price_step = price_step  # Step size for price adjustments
        self.reset()

    def _simulate_sales(self, price):
        """Simulate sales based on the price using the trained model."""
        # Generate dynamic features for prediction
        input_features = np.mean(X_imputed, axis=0)  # Use historical average values
        input_features[-1] = price  # Set the current price in the features
        return self.model.predict([input_features])[0]

    def reset(self):
        """Reset the environment to an initial state."""
        self.current_price = np.random.uniform(*self.price_range)
        simulated_sales = self._simulate_sales(self.current_price)
        self.state = (self.current_price, simulated_sales)
        return self.state

    def step(self, action):
        """
        Apply the action, calculate reward, and return the new state.
        Actions:
        - 0: Decrease price
        - 1: Keep price
        - 2: Increase price
        """
        price_change = (action - 1) * self.price_step  # Map action to price adjustment
        self.current_price += price_change
        self.current_price = np.clip(self.current_price, *self.price_range)

        # Simulate sales with the adjusted price
        simulated_sales = self._simulate_sales(self.current_price)

        # Calculate reward (sales - penalty for large price changes)
        reward = simulated_sales - abs(price_change)

        # Update state
        self.state = (self.current_price, simulated_sales)
        return self.state, reward

# Initialize the environment
env = PricingEnv(sales_model)

# Step 3: Implement Q-Learning
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-value table

    def _get_state_key(self, state):
        """Convert continuous state into a discrete key for Q-table."""
        price, sales = state
        return (round(price, -1), round(sales, -1))  # Round to nearest 10 for simplicity

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        state_key = self._get_state_key(state)
        if np.random.rand() < self.epsilon or state_key not in self.q_table:
            return np.random.choice([0, 1, 2])  # Explore
        return np.argmax(self.q_table[state_key])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-values using the Q-learning formula."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(3)  # Initialize Q-values for actions

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(3)  # Initialize Q-values for next state

        # Q-learning update rule
        best_next_action = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[state_key][action]
        )

# Train the agent
agent = QLearningAgent(env)
episodes = 500

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(50):  # Limit steps per episode
        action = agent.choose_action(state)
        next_state, reward = env.step(action)
        agent.update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Test the trained agent
state = env.reset()
for step in range(10):
    action = agent.choose_action(state)
    state, reward = env.step(action)
    print(f"Step {step + 1}: Action = {action}, State = {state}, Reward = {reward}")
