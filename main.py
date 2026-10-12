import numpy as np
import matplotlib.pyplot as plt
from env import GridWorld
from train import train_qlearning, train_sarsa
from visualize import plot_policy, generate_gif

def smooth(x, w=50):
    return np.convolve(x, np.ones(w)/w, mode='valid')

if __name__ == "__main__":
    env = GridWorld(size=10)
    
    print("Entraînement Q-Learning...")
    agent_ql, r_ql = train_qlearning(env, n_episodes=5000)
    
    print("Entraînement SARSA...")
    agent_sa, r_sa = train_sarsa(env, n_episodes=5000)
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(r_ql), label="Q-Learning", lw=2)
    plt.plot(smooth(r_sa), label="SARSA", lw=2)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense cumulée (lissée)")
    plt.title("Convergence : Q-Learning vs SARSA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("learning_curve.png", dpi=150, bbox_inches='tight')
    plt.show()
    plot_policy(agent_ql.Q, env, "Q-Learning", "policy_ql.png")
    plot_policy(agent_sa.Q, env, "SARSA", "policy_sa.png")
    generate_gif(agent_ql.Q, env, "agent_ql.gif")
    generate_gif(agent_sa.Q, env, "agent_sa.gif")