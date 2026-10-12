import numpy as np
from tqdm import tqdm
from agents import QLearning, SARSA
from env import GridWorld

def train_qlearning(env,n_episodes=5000,max_steps=500):
    agent = QLearning(env.n_states,env.n_actions,epsilon=0.3)
    rewards = []

    for episode in tqdm(range(n_episodes), desc="Q-Learning"):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        total_reward = 0
        for _ in range(max_steps):
            a = agent.selectionner_action(s_idx)
            s_next,r,done = env.step(s, a)
            s_next_idx = env.state_to_idx(s_next)
            agent.update(s_idx,a,r,s_next_idx,done)
            s, s_idx = s_next, s_next_idx
            total_reward += r

            if done:
                break
        agent.decroissance_epsilon()
        rewards.append(total_reward)

    return agent, rewards


def train_sarsa(env,n_episodes=5000,max_steps=500):
    agent = SARSA(env.n_states,env.n_actions,epsilon=0.3)
    rewards = []

    for ep in tqdm(range(n_episodes), desc="SARSA"):
        s = env.reset()
        s_idx = env.state_to_idx(s)
        a = agent.selectionner_action(s_idx)
        total_reward = 0
        for _ in range(max_steps):
            s_next,r,done = env.step(s, a)
            s_next_idx = env.state_to_idx(s_next)
            a_next = agent.selectionner_action(s_next_idx)
            agent.update(s_idx,a,r,s_next_idx,a_next,done)
            s, s_idx, a = s_next, s_next_idx, a_next
            total_reward += r
            if done:
                break
        agent.decroissance_epsilon()
        rewards.append(total_reward)

    return agent, rewards


if __name__ == "__main__":
    env = GridWorld(10)
    print("Training...")
    print("\n" + "=="*18)
    print("Entrainement de l'agent du Q-Learning...")
    agent_qlearning, reawards_qlearning = train_qlearning(env=env)
    print("\n" + "=="*18)
    print("Entrainement de l'agent du SARSA...")
    agent_sarsa, reawards_sarsa = train_sarsa(env=env)
    print("\n" + "=="*18)
    print("Fin des deux entrainements")


    np.save("agent_qlearning.npy", agent_qlearning.Q)
    np.save("agent_sarsa.npy", agent_sarsa.Q)
    np.save("rewards_qlearning.npy", reawards_qlearning)
    np.save("rewards_sarsa.npy", reawards_sarsa)