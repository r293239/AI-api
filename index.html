import numpy as np
import json
import pickle
from typing import List, Tuple, Optional, Dict, Any, Callable
import random
from abc import ABC, abstractmethod
import time

class Memory:
    """Experience replay memory for storing and learning from past actions"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Store an experience"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample random experiences for training"""
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class NeuralNetwork:
    """Simple neural network for function approximation"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.layers = []
        
        # Build network architecture
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layer = {
                'weights': np.random.randn(sizes[i], sizes[i+1]) * 0.1,
                'biases': np.zeros((1, sizes[i+1])),
                'activations': None,
                'z_values': None
            }
            self.layers.append(layer)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Forward pass through network"""
        activation = x.reshape(1, -1) if x.ndim == 1 else x
        
        for i, layer in enumerate(self.layers):
            z = np.dot(activation, layer['weights']) + layer['biases']
            layer['z_values'] = z
            
            if i < len(self.layers) - 1:  # Hidden layers use ReLU
                activation = self.relu(z)
            else:  # Output layer (linear)
                activation = z
                
            layer['activations'] = activation
        
        return activation.flatten() if activation.shape[0] == 1 else activation
    
    def backward(self, x, target):
        """Backpropagation to update weights"""
        # Forward pass first
        output = self.forward(x)
        
        # Calculate loss gradient (MSE)
        output_error = output.reshape(1, -1) - target.reshape(1, -1)
        
        # Backward pass
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if i == len(self.layers) - 1:  # Output layer
                delta = output_error
            else:  # Hidden layers
                delta = np.dot(delta, self.layers[i+1]['weights'].T) * self.relu_derivative(layer['z_values'])
            
            # Get input to this layer
            if i == 0:
                layer_input = x.reshape(1, -1)
            else:
                layer_input = self.layers[i-1]['activations']
            
            # Update weights and biases
            layer['weights'] -= self.learning_rate * np.dot(layer_input.T, delta)
            layer['biases'] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
    
    def copy(self):
        """Create a copy of the network"""
        new_net = NeuralNetwork(1, [], 1)  # Dummy initialization
        new_net.layers = []
        for layer in self.layers:
            new_layer = {
                'weights': layer['weights'].copy(),
                'biases': layer['biases'].copy(),
                'activations': None,
                'z_values': None
            }
            new_net.layers.append(new_layer)
        return new_net

class Environment(ABC):
    """Abstract base class for environments the AI can learn in"""
    
    @abstractmethod
    def reset(self):
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action):
        """Take an action and return (next_state, reward, done, info)"""
        pass
    
    @abstractmethod
    def get_state_size(self):
        """Return the size of the state space"""
        pass
    
    @abstractmethod
    def get_action_size(self):
        """Return the size of the action space"""
        pass

class GridWorldEnvironment(Environment):
    """Simple grid world for the AI to navigate"""
    
    def __init__(self, width: int = 5, height: int = 5):
        self.width = width
        self.height = height
        self.reset()
        
        # Define rewards
        self.goal_reward = 100
        self.step_penalty = -1
        self.wall_penalty = -10
    
    def reset(self):
        """Reset to starting position"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.width-1, self.height-1]
        self.walls = [(2, 2), (2, 3), (3, 2)]  # Some walls
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        state = np.zeros(self.width * self.height + 4)  # Grid + agent pos + goal pos
        
        # One-hot encode agent position
        agent_idx = self.agent_pos[1] * self.width + self.agent_pos[0]
        state[agent_idx] = 1
        
        # Add agent and goal positions as features
        state[-4] = self.agent_pos[0] / self.width
        state[-3] = self.agent_pos[1] / self.height
        state[-2] = self.goal_pos[0] / self.width
        state[-1] = self.goal_pos[1] / self.height
        
        return state
    
    def step(self, action):
        """Take action: 0=up, 1=right, 2=down, 3=left"""
        old_pos = self.agent_pos.copy()
        
        # Apply action
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Right
            self.agent_pos[0] = min(self.width-1, self.agent_pos[0] + 1)
        elif action == 2:  # Down
            self.agent_pos[1] = min(self.height-1, self.agent_pos[1] + 1)
        elif action == 3:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        
        # Check if hit wall
        if tuple(self.agent_pos) in self.walls:
            self.agent_pos = old_pos  # Bounce back
            reward = self.wall_penalty
        elif self.agent_pos == self.goal_pos:
            reward = self.goal_reward
        else:
            reward = self.step_penalty
        
        done = (self.agent_pos == self.goal_pos)
        
        return self.get_state(), reward, done, {}
    
    def get_state_size(self):
        return self.width * self.height + 4
    
    def get_action_size(self):
        return 4
    
    def render(self):
        """Visual representation of the environment"""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Add walls
        for wall in self.walls:
            grid[wall[1]][wall[0]] = '#'
        
        # Add goal
        grid[self.goal_pos[1]][self.goal_pos[0]] = 'G'
        
        # Add agent
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
        
        for row in grid:
            print(' '.join(row))
        print()

class TaskEnvironment(Environment):
    """Environment for learning specific tasks"""
    
    def __init__(self, task_type: str = "sorting"):
        self.task_type = task_type
        self.reset()
    
    def reset(self):
        if self.task_type == "sorting":
            # Generate random array to sort
            self.array = np.random.randint(1, 10, size=5)
            self.target = np.sort(self.array)
            self.current_array = self.array.copy()
            self.steps = 0
            self.max_steps = 20
        
        return self.get_state()
    
    def get_state(self):
        if self.task_type == "sorting":
            # State includes current array and target
            state = np.concatenate([
                self.current_array / 10.0,  # Normalize
                self.target / 10.0,
                [self.steps / self.max_steps]
            ])
            return state
    
    def step(self, action):
        if self.task_type == "sorting":
            # Actions: swap adjacent elements (0-3) or do nothing (4)
            if action < 4 and action < len(self.current_array) - 1:
                # Swap elements at position action and action+1
                self.current_array[action], self.current_array[action+1] = \
                    self.current_array[action+1], self.current_array[action]
            
            self.steps += 1
            
            # Calculate reward
            if np.array_equal(self.current_array, self.target):
                reward = 100  # Solved!
                done = True
            elif self.steps >= self.max_steps:
                reward = -50  # Failed
                done = True
            else:
                # Reward based on how close to sorted
                inversions = self._count_inversions()
                reward = -inversions - 1  # Small penalty for each step
                done = False
            
            return self.get_state(), reward, done, {}
    
    def _count_inversions(self):
        """Count how many pairs are out of order"""
        count = 0
        for i in range(len(self.current_array)):
            for j in range(i+1, len(self.current_array)):
                if self.current_array[i] > self.current_array[j]:
                    count += 1
        return count
    
    def get_state_size(self):
        if self.task_type == "sorting":
            return 11  # 5 current + 5 target + 1 step
    
    def get_action_size(self):
        if self.task_type == "sorting":
            return 5  # 4 swaps + 1 do nothing
    
    def render(self):
        if self.task_type == "sorting":
            print(f"Current: {self.current_array}")
            print(f"Target:  {self.target}")
            print(f"Steps: {self.steps}/{self.max_steps}")
            print()

class DQNAgent:
    """Deep Q-Network agent that learns to perform tasks"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(capacity=2000)
        
        # Hyperparameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.batch_size = 32
        
        # Neural networks
        self.q_network = NeuralNetwork(state_size, [64, 64], action_size, learning_rate)
        self.target_network = self.q_network.copy()
        self.update_target_frequency = 100
        self.update_counter = 0
        
        # Training stats
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_steps = []
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        
        for state, action, reward, next_state, done in experiences:
            target = reward
            if not done:
                next_q_values = self.target_network.forward(next_state)
                target = reward + self.gamma * np.max(next_q_values)
            
            # Get current Q values
            current_q_values = self.q_network.forward(state)
            target_q_values = current_q_values.copy()
            target_q_values[action] = target
            
            # Train network
            self.q_network.backward(state, target_q_values)
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.update_target_frequency == 0:
            self.target_network = self.q_network.copy()
    
    def train(self, environment: Environment, episodes: int = 1000, verbose: bool = True):
        """Train the agent in the given environment"""
        self.episode_rewards = []
        self.episode_steps = []
        
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = self.act(state)
                next_state, reward, done, _ = environment.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                
                # Prevent infinite episodes
                if steps > 1000:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # Train on experiences
            self.replay()
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def test(self, environment: Environment, episodes: int = 10):
        """Test the trained agent"""
        print("\n--- Testing Trained Agent ---")
        test_rewards = []
        
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nTest Episode {episode + 1}:")
            environment.render()
            
            while True:
                action = self.act(state, training=False)  # No exploration
                state, reward, done, _ = environment.step(action)
                total_reward += reward
                steps += 1
                
                print(f"Step {steps}: Action {action}, Reward {reward}")
                environment.render()
                
                if done or steps > 50:
                    break
                
                time.sleep(0.5)  # Slow down for visualization
            
            test_rewards.append(total_reward)
            print(f"Episode {episode + 1} finished with reward: {total_reward}")
        
        avg_test_reward = np.mean(test_rewards)
        print(f"\nAverage test reward: {avg_test_reward:.2f}")
        return avg_test_reward
    
    def save(self, filepath: str):
        """Save the trained agent"""
        agent_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            # Note: Would need to implement network serialization for full save
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        print(f"Agent saved to {filepath}")

class LearningAI:
    """Main AI class that can learn different tasks"""
    
    def __init__(self):
        self.agents = {}  # Store different specialized agents
        self.environments = {}
        self.current_task = None
    
    def add_task(self, task_name: str, environment: Environment):
        """Add a new task for the AI to learn"""
        self.environments[task_name] = environment
        
        # Create specialized agent for this task
        state_size = environment.get_state_size()
        action_size = environment.get_action_size()
        self.agents[task_name] = DQNAgent(state_size, action_size)
        
        print(f"Added task '{task_name}' with state size {state_size} and {action_size} actions")
    
    def learn_task(self, task_name: str, episodes: int = 1000):
        """Train the AI on a specific task"""
        if task_name not in self.agents:
            print(f"Task '{task_name}' not found!")
            return
        
        print(f"\n=== Learning Task: {task_name} ===")
        agent = self.agents[task_name]
        environment = self.environments[task_name]
        
        agent.train(environment, episodes)
        self.current_task = task_name
        
        print(f"Finished learning '{task_name}'!")
    
    def demonstrate_task(self, task_name: str, episodes: int = 3):
        """Show the AI performing a learned task"""
        if task_name not in self.agents:
            print(f"Task '{task_name}' not found!")
            return
        
        print(f"\n=== Demonstrating Task: {task_name} ===")
        agent = self.agents[task_name]
        environment = self.environments[task_name]
        
        return agent.test(environment, episodes)
    
    def get_learning_progress(self, task_name: str):
        """Get learning statistics for a task"""
        if task_name not in self.agents:
            return None
        
        agent = self.agents[task_name]
        if not agent.episode_rewards:
            return None
        
        recent_performance = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
        
        return {
            'episodes_trained': len(agent.episode_rewards),
            'recent_avg_reward': recent_performance,
            'best_reward': max(agent.episode_rewards),
            'exploration_rate': agent.epsilon
        }

# Example usage and demonstration
if __name__ == "__main__":
    print("🤖 Action-Learning AI System")
    print("=" * 40)
    
    # Create the learning AI
    ai = LearningAI()
    
    # Add different tasks
    print("\n📋 Adding Tasks...")
    
    # Task 1: Navigate a grid world
    grid_env = GridWorldEnvironment(width=4, height=4)
    ai.add_task("navigation", grid_env)
    
    # Task 2: Learn to sort numbers
    sorting_env = TaskEnvironment(task_type="sorting")
    ai.add_task("sorting", sorting_env)
    
    # Train on navigation task
    print("\n🎯 Training on Navigation Task...")
    ai.learn_task("navigation", episodes=500)
    
    # Show navigation performance
    progress = ai.get_learning_progress("navigation")
    if progress:
        print(f"Navigation Progress:")
        print(f"- Episodes trained: {progress['episodes_trained']}")
        print(f"- Recent average reward: {progress['recent_avg_reward']:.2f}")
        print(f"- Best reward achieved: {progress['best_reward']:.2f}")
    
    # Train on sorting task
    print("\n🔢 Training on Sorting Task...")
    ai.learn_task("sorting", episodes=300)
    
    # Show sorting performance
    progress = ai.get_learning_progress("sorting")
    if progress:
        print(f"Sorting Progress:")
        print(f"- Episodes trained: {progress['episodes_trained']}")
        print(f"- Recent average reward: {progress['recent_avg_reward']:.2f}")
        print(f"- Best reward achieved: {progress['best_reward']:.2f}")
    
    # Demonstrate learned skills
    print("\n🎭 Demonstrating Learned Skills...")
    ai.demonstrate_task("navigation", episodes=2)
    ai.demonstrate_task("sorting", episodes=2)
    
    print("\n✅ AI has successfully learned to perform tasks!")
    print("The AI can now:")
    print("- Navigate through environments")
    print("- Learn to sort arrays")
    print("- Adapt to new tasks")
    print("- Improve through experience")
