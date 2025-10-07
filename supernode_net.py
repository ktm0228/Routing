
import math
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Node:
    """Single network node with position, battery, role, and neighbor list."""
    id: int
    x: float
    y: float
    battery: float = 1.0
    is_supernode: bool = False
    neighbors: List[int] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)


class Network:
    """Ad-hoc network model with deterministic routing and simple visuals.

    Notes:
        - No external styling; Matplotlib defaults only (per instructions).
        - One chart per figure.
        - Methods are written to be extended later with MAC models or ML modules.
    """
    def __init__(self, L: float = 1.0):
        self.L = L  # field side length
        self.nodes: Dict[int, Node] = {}

    # ---------- Construction ----------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def connect_nodes(self, i: int, j: int) -> None:
        if j not in self.nodes[i].neighbors:
            self.nodes[i].neighbors.append(j)
        if i not in self.nodes[j].neighbors:
            self.nodes[j].neighbors.append(i)

    def connect_by_radius(self, radius: float) -> None:
        ids = list(self.nodes.keys())
        for a in range(len(ids)):
            i = ids[a]
            xi, yi = self.nodes[i].pos()
            for b in range(a + 1, len(ids)):
                j = ids[b]
                xj, yj = self.nodes[j].pos()
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= radius ** 2:
                    self.connect_nodes(i, j)

    # ---------- Utilities ----------
    def euclid(self, i: int, j: int) -> float:
        xi, yi = self.nodes[i].pos()
        xj, yj = self.nodes[j].pos()
        return math.hypot(xi - xj, yi - yj)

    def degree(self, i: int) -> int:
        return len(self.nodes[i].neighbors)

    def pick_supernodes(self, frac: float = 0.1, mode: str = "degree") -> List[int]:
        """Select supernodes by top battery or top degree and set flags."""
        n = len(self.nodes)
        k = max(1, int(round(frac * n)))
        ids = list(self.nodes.keys())
        if mode == "battery":
            ids_sorted = sorted(ids, key=lambda i: self.nodes[i].battery, reverse=True)
        elif mode == "degree":
            ids_sorted = sorted(ids, key=lambda i: self.degree(i), reverse=True)
        else:
            ids_sorted = ids[:]  # fallback: take first k
        chosen = ids_sorted[:k]
        for i in self.nodes:
            self.nodes[i].is_supernode = (i in chosen)
        return chosen

    # ---------- Routing ----------
    def bfs_shortest_path(self, s: int, t: int) -> List[int]:
        """Breadth-first search for minimum-hop path."""
        if s == t:
            return [s]
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in self.nodes[u].neighbors:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            return []
        # reconstruct path
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def greedy_route(self, s: int, t: int, max_hops: Optional[int] = None) -> List[int]:
        """Greedy geographic routing with simple backtracking to avoid dead-ends."""
        if s == t:
            return [s]
        visited = {s}
        path = [s]
        current = s
        hops_limit = max_hops or (len(self.nodes) + 5)
        for _ in range(hops_limit):
            nbrs = [v for v in self.nodes[current].neighbors if v not in visited]
            if not nbrs:
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    break
            next_hop = min(nbrs, key=lambda v: self.euclid(v, t))
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            if current == t:
                return path
        return path if path and path[-1] == t else []

    def supernode_route(self, s: int, t: int) -> List[int]:
        """Three-stage route via supernode backbone (fallbacks if needed)."""
        super_ids = [i for i, n in self.nodes.items() if n.is_supernode]
        if not super_ids:
            return self.bfs_shortest_path(s, t)
        s_star = min(super_ids, key=lambda i: self.euclid(s, i))
        t_star = min(super_ids, key=lambda i: self.euclid(t, i))
        # stage 1
        p1 = self.bfs_shortest_path(s, s_star)
        if not p1:
            return []
        # stage 2: BFS on supernode-only subgraph
        super_adj = defaultdict(list)
        for i in super_ids:
            for j in self.nodes[i].neighbors:
                if self.nodes[j].is_supernode:
                    super_adj[i].append(j)
        prev = {s_star: None}
        q = deque([s_star])
        while q:
            u = q.popleft()
            if u == t_star:
                break
            for v in super_adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t_star in prev:
            p2 = []
            cur = t_star
            while cur is not None:
                p2.append(cur)
                cur = prev[cur]
            p2 = list(reversed(p2))
        else:
            # fallback to BFS on the full graph between supernodes
            p2 = self.bfs_shortest_path(s_star, t_star)
            if not p2:
                return []
        # stage 3
        p3 = self.bfs_shortest_path(t_star, t)
        if not p3:
            return []
        # merge
        return p1[:-1] + p2[:-1] + p3

    # ---------- Visualization ----------
    def plot_topology(self, title: str = "Network Topology", show_ids: bool = False, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(7, 6))
        # edges
        for i, node in self.nodes.items():
            xi, yi = node.pos()
            for j in node.neighbors:
                if j > i:
                    xj, yj = self.nodes[j].pos()
                    ax.plot([xi, xj], [yi, yj])
        # nodes
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        ax.scatter(xs, ys, s=20)
        # supernodes
        sx = [n.x for n in self.nodes.values() if n.is_supernode]
        sy = [n.y for n in self.nodes.values() if n.is_supernode]
        if len(sx) > 0:
            ax.scatter(sx, sy, s=80, marker='s')
        if show_ids and len(self.nodes) <= 200:
            for i, n in self.nodes.items():
                ax.text(n.x, n.y, str(i), fontsize=7)
        ax.set_title(title)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_route(self, path: List[int], title: str = "Route", save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(7, 6))
        # base edges
        for i, node in self.nodes.items():
            xi, yi = node.pos()
            for j in node.neighbors:
                if j > i:
                    xj, yj = self.nodes[j].pos()
                    ax.plot([xi, xj], [yi, yj], linewidth=0.8)
        # nodes
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        ax.scatter(xs, ys, s=20)
        # supernodes
        sx = [n.x for n in self.nodes.values() if n.is_supernode]
        sy = [n.y for n in self.nodes.values() if n.is_supernode]
        if len(sx) > 0:
            ax.scatter(sx, sy, s=80, marker='s')
        # route overlay
        if path and len(path) > 1:
            for u, v in zip(path[:-1], path[1:]):
                x1, y1 = self.nodes[u].pos()
                x2, y2 = self.nodes[v].pos()
                ax.plot([x1, x2], [y1, y2], linewidth=3.0)
        ax.set_title(title)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ---------- Helpers for random networks ----------
def suggest_radius(n: int, L: float, safety: float = 1.5) -> float:
    """Heuristic connection radius for random geometric graphs to be likely connected."""
    if n < 5:
        return 0.5 * L
    return safety * L * math.sqrt(max(1e-9, math.log(n) / (math.pi * n)))

def make_random_network(n: int, L: float, radius: Optional[float] = None, seed: Optional[int] = None,
                        super_frac: float = 0.1, super_mode: str = "degree") -> Network:
    rng = np.random.default_rng(seed)
    net = Network(L=L)
    coords = rng.random((n, 2)) * L
    batteries = 0.5 + 0.5 * rng.random(n)  # random in [0.5, 1.0]
    for i in range(n):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        net.add_node(Node(i, x, y, battery=float(batteries[i])))
    R = radius if radius is not None else suggest_radius(n, L)
    net.connect_by_radius(R)
    net.pick_supernodes(frac=super_frac, mode=super_mode)
    return net
