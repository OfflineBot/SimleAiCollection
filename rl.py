import numpy as np

grid = np.array([
    ['P', ' ', ' ', ' ', ' ', ' '],
    [' ', 'X', 'X', 'X', 'X', ' '],
    [' ', ' ', ' ', 'X', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', 'X', 'X', ' ', ' '],
    ['X', ' ', ' ', ' ', ' ', 'G'],
    [' ', ' ', ' ', ' ', ' ', ' ']
])

num_states = grid.shape[0] * grid.shape[1]
num_actions = 4  # Up, Down, Left, Right
q_table = np.zeros((num_states, num_actions))

learning_rate = 0.1
discount_factor = 0.9
num_episodes = 20_000
max_steps_per_episode = 100

for episode in range(num_episodes):
    player_pos = np.where(grid == 'P')
    state = player_pos[0] * grid.shape[1] + player_pos[1]

    if episode % 1_000 == 0:
        print(episode / num_episodes * 100)

    for step in range(max_steps_per_episode):
        if np.random.uniform(0, 1) < 0.1:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])

        row, col = divmod(state, grid.shape[1])

        if action == 0:  # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            new_state = state
        else:
            new_state = row * grid.shape[1] + col

        if grid.flat[new_state] == 'X':
            reward = -100
        elif grid.flat[new_state] == 'G':  # Goal reached
            reward = 10
        else:
            reward = -1

        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

        if episode == num_episodes - 1:
            print("x: ", row, " | y: ", col)
            if grid.flat[state] == 'G':
                print("finished")

        if grid.flat[state] == 'G':  # Goal reached
            break

print("Learned Q-table:")
print(q_table)