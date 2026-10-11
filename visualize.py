import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from env import GridWorld


def plot_policy(Q, env, title="Politique apprise", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    policy = np.argmax(Q, axis=1).reshape(env.size, env.size)
    arrows = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}

    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))
            elif (i, j) == env.goal:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='gold'))
            elif (i, j) == env.start:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgreen'))
            else:
                a = policy[i, j]
                dx, dy = arrows[a]
                ax.arrow(j, i, dx, dy, head_width=0.15, color='steelblue')

    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_to_rgb_array(fig):
    fig.canvas.draw()
    try:
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        try:
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, 1:]
    return image


def generate_gif(Q, env, path="agent.gif", max_steps=100, fps=5):
    frames = []
    s = env.start
    s_idx = env.state_to_idx(s)
    last_image = None

    for step in range(max_steps):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(-0.5, env.size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for (i, j) in env.obstacles:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))

        ax.add_patch(plt.Rectangle(
            (env.goal[1] - 0.5, env.goal[0] - 0.5), 1, 1, color='gold'
        ))

        ax.plot(s[1], s[0], 'o', markersize=20, color='red', label='Agent')
        ax.set_title(f"Épisode optimal — Étape {step}")
        ax.legend(loc='upper left')

        image = fig_to_rgb_array(fig)
        frames.append(image)
        last_image = image
        plt.close(fig)

        a = int(np.argmax(Q[s_idx]))
        s_next, r, done = env.step(s, a)
        s, s_idx = s_next, env.state_to_idx(s_next)

        if done:
            for _ in range(10):
                frames.append(last_image)
            break

    imageio.mimsave(path, frames, fps=fps)
    print(f"GIF sauvegardé : {path} ({len(frames)} frames)")


if __name__ == "__main__":
    env = GridWorld(size=10)
    Q_ql = np.load("agent_qlearning.npy")
    Q_sa = np.load("agent_sarsa.npy")

    plot_policy(Q_ql, env, "Q-Learning", "policy_ql.png")
    plot_policy(Q_sa, env, "SARSA", "policy_sa.png")

    generate_gif(Q_ql, env, "agent_ql.gif")
    generate_gif(Q_sa, env, "agent_sa.gif")
