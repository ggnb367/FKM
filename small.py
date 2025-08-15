import random
import time
import heapq
from collections import defaultdict

import networkx as nx
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import community.community_louvain as community_louvain


class PrunedPLLIndex:
    """Pruned Landmark Labeling for weighted graphs."""

    def __init__(self, G: nx.Graph, order=None):
        self.G = G
        self.labels = {v: {} for v in G.nodes()}
        self.order = list(order) if order is not None else list(G.nodes())

    def build(self):
        for v in tqdm(self.order, desc="Building PLL index", unit="node"):
            self._pruned_dijkstra(v)

    def _pruned_dijkstra(self, root):
        dist = {root: 0}
        heap = [(0, 0, root)]
        counter = 0
        while heap:
            d, _, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if self.query(root, u) <= d:
                continue
            self.labels[u][root] = d
            for v, data in self.G[u].items():
                w = data.get("weight", 1)
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    counter += 1
                    heapq.heappush(heap, (nd, counter, v))

    def query(self, u, v):
        best = float("inf")
        labels_u = self.labels.get(u, {})
        labels_v = self.labels.get(v, {})
        if len(labels_u) > len(labels_v):
            labels_u, labels_v = labels_v, labels_u
        for landmark, du in labels_u.items():
            dv = labels_v.get(landmark)
            if dv is not None:
                total = du + dv
                if total < best:
                    best = total
        return best


def load_cora():
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    g = nx.Graph()
    for u, v in zip(edge_index[0], edge_index[1]):
        g.add_edge(int(u), int(v), weight=1)
    return g


def build_hybrid_index(G, partition):
    clusters = defaultdict(list)
    for node, cid in partition.items():
        clusters[cid].append(node)
    primary = {cid: nodes for cid, nodes in clusters.items() if len(nodes) > 25}

    inside_pll = {}
    boundary_sets = {}
    boundary_pairs = {}
    node_cluster = {}

    for cid, nodes in primary.items():
        subg = G.subgraph(nodes).copy()
        ecc = {}
        for comp in nx.connected_components(subg):
            ecc.update(nx.eccentricity(subg.subgraph(comp)))
        order = sorted(nodes, key=lambda n: ecc[n])
        pll = PrunedPLLIndex(subg, order)
        pll.build()
        inside_pll[cid] = pll
        boundary = {n for n in nodes if any(neigh not in nodes for neigh in G.neighbors(n))}
        boundary_sets[cid] = boundary
        node_cluster.update({n: cid for n in nodes})
        pairs = {}
        b_list = list(boundary)
        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                u, v = b_list[i], b_list[j]
                pairs[(u, v)] = pll.query(u, v)
        boundary_pairs[cid] = pairs

    outside_nodes = set()
    for cid in primary:
        outside_nodes |= boundary_sets[cid]
    for cid, nodes in clusters.items():
        if cid not in primary:
            outside_nodes |= set(nodes)
            node_cluster.update({n: None for n in nodes})

    G_out = nx.Graph()
    G_out.add_nodes_from(outside_nodes)
    for u, v, data in G.edges(data=True):
        if u in outside_nodes and v in outside_nodes:
            G_out.add_edge(u, v, weight=data.get("weight", 1))
    for pairs in boundary_pairs.values():
        for (u, v), d in pairs.items():
            if G_out.has_edge(u, v):
                if d < G_out[u][v].get("weight", float("inf")):
                    G_out[u][v]["weight"] = d
            else:
                G_out.add_edge(u, v, weight=d)

    # Order outside graph nodes by degree (unweighted)
    degree_order = sorted(
        G_out.degree(), key=lambda x: x[1], reverse=True
    )
    order = [n for n, _ in degree_order]
    outside_pll = PrunedPLLIndex(G_out, order)
    outside_pll.build()

    return inside_pll, boundary_sets, outside_pll, node_cluster


def query_hybrid(u, v, node_cluster, inside_pll, boundary_sets, outside_pll):
    def project(x):
        cid = node_cluster.get(x)
        if cid is None:
            return x, 0
        boundary = boundary_sets[cid]
        if x in boundary:
            return x, 0
        pll = inside_pll[cid]
        best = (float("inf"), None)
        for b in boundary:
            d = pll.query(x, b)
            if d < best[0]:
                best = (d, b)
        return best[1], best[0]

    e_u, d_u = project(u)
    e_v, d_v = project(v)
    mid = outside_pll.query(e_u, e_v) if e_u != e_v else 0
    return d_u + mid + d_v


def evaluate(G):
    partition = community_louvain.best_partition(G)

    pll = PrunedPLLIndex(G)
    start = time.time()
    pll.build()
    pll_time = time.time() - start

    start = time.time()
    inside_pll, boundary_sets, outside_pll, node_cluster = build_hybrid_index(G, partition)
    hybrid_time = time.time() - start

    nodes = list(G.nodes())
    pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(1000)]

    # Precompute ground-truth distances before timing the queries
    gt_cache = []
    for u, v in pairs:
        try:
            d = nx.shortest_path_length(G, u, v, weight="weight")
            gt_cache.append((u, v, d))
        except Exception:
            continue

    result = []
    for name, func in [
        ("PLL", lambda u, v: pll.query(u, v)),
        ("Hybrid", lambda u, v: query_hybrid(u, v, node_cluster, inside_pll, boundary_sets, outside_pll)),
    ]:
        correct = 0
        total = 0
        mae = 0.0
        start = time.time()
        for u, v, gt in gt_cache:
            est = func(u, v)
            if est == gt:
                correct += 1
            if est != float("inf"):
                mae += abs(est - gt)
                total += 1
        end = time.time()
        result.append({
            "method": name,
            "query_time_sec": end - start,
            "mae": mae / total if total > 0 else float("inf"),
            "samples": total,
            "exact_matches": correct,
            "build_time_sec": pll_time if name == "PLL" else hybrid_time,
        })
    return pd.DataFrame(result)


def main():
    G = load_cora()
    df = evaluate(G)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
