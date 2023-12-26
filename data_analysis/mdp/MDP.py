class Node:
    def __init__(self, state):
        self.state = state
        self.edges = set()
        # for column in columns:
        #     self.state.append(column)

    def add_edge(self, next_node, action, reward, done, cost, weight, prob):
        self.edges.add(Edge(next_node, action, reward, done, cost, weight, prob))

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __hash__(self):
        return hash(id(self.state))


class Edge:
    def __init__(self, next_node, action, reward, done, cost, weight, prob):
        self.next_node = next_node
        self.action = action
        self.reward = reward
        self.done = done
        self.cost = cost
        self.weight = weight
        self.prob = prob

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.next_node == other.next_node
        return False

    def __hash__(self):
        return hash((
            hash(id(self.next_node)),
            tuple(self.action),
            tuple(self.reward),
            self.done,
            tuple(self.cost),
            self.weight,
            self.prob
        ))


class Attr:
    def __init__(self, state):
        self.state = state
        self.probs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = []
        self.weights = []
        self.next_states = []
        # TODO config内的action的action_up_bound,action_low_bound
        self.max_action = -100
        self.min_action = 100
        for edge in state.edges:
            self.probs.append(edge.prob)
            self.actions.append(edge.action)
            self.done.append(edge.done)
            self.weights.append(edge.weight)
            self.rewards.append(edge.reward)
            self.next_states.append(edge.next_node)
            self.costs.append(edge.cost)
            # TODO 最大最小需要根据动作维度确定，这里先用一维
            self.min_action = min(self.min_action, edge.action[0])
            self.max_action = max(self.max_action, edge.action[0])
