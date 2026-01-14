import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Grid World Definition
# -------------------------
rows, cols = 5, 5
goal_state = (4, 4)
negative_state = (3, 3)
obstacles = [(1, 1)]
states = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in obstacles]

ACTIONS = ['U', 'D', 'L', 'R']

# Reward Function
def reward(state):
    if state == goal_state:
        return 10
    if state == negative_state:
        return -10
    return -0.1

# Movement Function
def move(state, action):
    r, c = state
    if action == 'U': r -= 1
    elif action == 'D': r += 1
    elif action == 'L': c -= 1
    elif action == 'R': c += 1
    next_state = (r, c)
    if (r < 0 or r >= rows or c < 0 or c >= cols or next_state in obstacles):
        return state
    return next_state

# Stochastic Transitions
def get_next_states(state, action):
    transitions = []
    intended = move(state, action)
    transitions.append((0.8, intended))
    other_actions = [a for a in ACTIONS if a != action]
    prob = 0.2 / len(other_actions)
    for a in other_actions:
        transitions.append((prob, move(state, a)))
    return transitions

# Value Iteration
def value_iteration(gamma=0.9, theta=0.001):
    V = {s:0 for s in states}
    while True:
        delta = 0
        for s in states:
            if s in [goal_state, negative_state]: continue
            values = []
            for a in ACTIONS:
                total = 0
                for prob, s_next in get_next_states(s, a):
                    total += prob * (reward(s_next) + gamma * V[s_next])
                values.append(total)
            best_value = max(values)
            delta = max(delta, abs(V[s]-best_value))
            V[s] = best_value
        if delta < theta: break
    return V

# Extract Policy
def extract_policy(V, gamma):
    policy = {}
    for s in states:
        if s in [goal_state, negative_state]: policy[s]='T'; continue
        best_action = None; best_value=-1e9
        for a in ACTIONS:
            total=0
            for prob, s_next in get_next_states(s,a):
                total += prob*(reward(s_next)+gamma*V[s_next])
            if total>best_value: best_value=total; best_action=a
        policy[s]=best_action
    return policy

# Policy Iteration
def policy_iteration(gamma=0.9):
    policy = {s: np.random.choice(ACTIONS) for s in states}
    V = {s:0 for s in states}
    stable = False
    while not stable:
        # Policy Evaluation
        for _ in range(20):
            for s in states:
                if s in [goal_state, negative_state]: continue
                a=policy[s]
                total=0
                for prob, s_next in get_next_states(s,a):
                    total+=prob*(reward(s_next)+gamma*V[s_next])
                V[s]=total
        # Policy Improvement
        stable=True
        for s in states:
            if s in [goal_state, negative_state]: continue
            best_action=extract_policy(V,gamma)[s]
            if best_action!=policy[s]:
                policy[s]=best_action
                stable=False
    return policy, V

# Visualization
def plot_values(V):
    grid=np.zeros((rows,cols))
    for (r,c),v in V.items(): grid[r,c]=v
    fig,ax=plt.subplots()
    im=ax.imshow(grid,cmap='coolwarm')
    plt.colorbar(im)
    ax.set_title("Value Function")
    return fig

def plot_policy(policy):
    fig,ax=plt.subplots()
    ax.set_xlim(-0.5,cols-0.5)
    ax.set_ylim(rows-0.5,-0.5)
    ax.set_title("Policy")
    for r in range(rows):
        for c in range(cols):
            if (r,c) in obstacles: ax.text(c,r,"X",ha='center',va='center')
            elif (r,c)==goal_state: ax.text(c,r,"G",ha='center',va='center')
            elif (r,c)==negative_state: ax.text(c,r,"N",ha='center',va='center')
            else: a=policy.get((r,c),''); ax.text(c,r,a,ha='center',va='center')
    ax.grid(True)
    return fig

# Streamlit UI
st.title("Grid World MDP – Value & Policy Iteration")
algorithm = st.selectbox("Select Algorithm",["Value Iteration","Policy Iteration"])
gamma = st.slider("Discount Factor (γ)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

if st.button("Run Algorithm"):
    if algorithm=="Value Iteration":
        V=value_iteration(gamma)
        policy=extract_policy(V,gamma)
    else:
        policy,V=policy_iteration(gamma)
    st.subheader("Value Function")
    st.pyplot(plot_values(V))
    st.subheader("Optimal Policy")
    st.pyplot(plot_policy(policy))
