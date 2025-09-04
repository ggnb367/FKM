# Prompt for Codex — Hierarchical Leiden Hybrid Shortest Paths (Colab CPU)

**Important:** Output **code only** (no explanations). Produce a single Python file named `hybrid_hier_leiden.py` that runs in **Google Colab CPU-only** (no CUDA). The script must build and compare three shortest-path query methods on a Planetoid graph (default **PubMed**), using **graspologic.partition.hierarchical_leiden** for hierarchical community detection with **modularity**. **Do NOT implement any “small-cluster dissolve” step.**

At the end, print a `pandas.DataFrame` **with exactly these four columns (in this order)**:
```
method  query_time_sec  build_time_sec  mae
```
(no index). If saving to CSV is requested, save **only** these four columns.

---

## Colab CPU-only install (comment at top of the script)

```bash
# pip install torch torchvision torchaudio cpuonly -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch-geometric
# pip install graspologic networkx pandas tqdm numpy numba scipy matplotlib
```

---

## Requirements & Structure (single file, type-annotated, docstrings, clear comments)

### 1) Data Loading

- Function: `load_planetoid_graph(name: str = "PubMed") -> nx.Graph`
  - Use `torch_geometric.datasets.Planetoid(root="/tmp/Planetoid", name=name)`.
  - Accept `"Pubmed"` as alias to `"PubMed"`. Also support `"Cora"`, `"CiteSeer"`.
  - Convert `edge_index` to an **undirected** `networkx.Graph`, set all edge weights to `1`, keep original integer node ids.

### 2) Hierarchical Community Detection (graspologic)

- Function:
```python
def run_hierarchical_leiden(
    G: nx.Graph,
    max_cluster_size: int = 1000,
    resolution: float = 1.0,
    randomness: float = 1e-3,
    seed: int = 42,
) -> Tuple[
    Dict[int, Dict[int, List[int]]],          # levels[level][cluster_id] = [nodes...]
    Dict[int, Dict[int, int]],                # node2level2cluster[node][level] = cluster_id
    Dict[Tuple[int, int], Optional[int]],     # parent_of[(child_level, child_cid)] = parent_cid  (parent level = child_level-1)
    Dict[Tuple[int, int], List[int]],         # children_of[(parent_level, parent_cid)] = [child_cids...]  (children at parent_level+1)
    int,                                      # root_level (= 0)
    int                                       # max_level (deepest level)
]
```
- Implementation:
  - Call:
    ```python
    from graspologic.partition import hierarchical_leiden
    hc = hierarchical_leiden(
        G,
        max_cluster_size=max_cluster_size,
        starting_communities=None,
        extra_forced_iterations=0,
        resolution=resolution,
        randomness=randomness,
        use_modularity=True,
        random_seed=seed,
        weight_attribute="weight",
        is_weighted=None,
        weight_default=1.0,
        check_directed=True,
    )
    ```
  - Build the outputs from `hc` entries (`node`, `cluster`, `parent_cluster`, `level`, `is_final_cluster`):
    - `levels[level][cid] = list of node ids`.
    - `node2level2cluster[node][level] = cid`.
    - `parent_of[(child_level, child_cid)] = parent_cid` (parent level = `child_level - 1`, parent_cid can be `None` when root).
    - `children_of[(parent_level, parent_cid)] = [child_cids...]`.
  - `root_level = 0`, `max_level = max(levels.keys())`. Handle single-level cases gracefully.

### 3) Intra-cluster PLL (for **every** cluster at **every** level)

- Class: `PrunedPLLIndex`
  - `__init__(G: nx.Graph, order: Optional[List[int]] = None)`
  - `build()` runs pruned Dijkstra per root following classic Pruned Landmark Labeling.
  - `query(u: int, v: int) -> float` uses label intersection; return `float("inf")` if unreachable.
- Function:
```python
def build_pll_for_all_clusters(
    G: nx.Graph,
    levels: Dict[int, Dict[int, List[int]]]
) -> Dict[Tuple[int, int], PrunedPLLIndex]:
    # Build PLL per (level, cluster_id). For each cluster subgraph:
    # - Compute eccentricity per connected component, then merge;
    # - Use nodes sorted by ascending eccentricity as build order;
    # - Store as pll[(level, cid)].
```

### 4) Parent-level “one-hop” inter-subcluster ALT indices

For **each parent cluster** at level `L` that has children at `L+1`:
1. Let `children = children_of[(L, parent_cid)]`.
2. For each child cluster `Ci`, define **boundary nodes**: nodes of `Ci` adjacent (in the full parent subgraph) to nodes in any **other** child `Cj` (same parent).
3. For each pair (`Ci`, `Cj`) with `i < j`, examine edges `(u, v)` with `u ∈ Ci`, `v ∈ Cj`. Keep the pair with **minimum weight** (tie-break by `(u, v)` lexicographically) as the **representative pair** for (`Ci`, `Cj`).
4. Build a **representative graph `G_rep`** for this parent:
   - Nodes: the set of all representative nodes across its children.
   - Edges:
     - **Intra-child**: fully connect representative nodes of the **same child cluster** with edge weight = **PLL[(L+1, child_cid)].query(u, v)**.
     - **Inter-child**: connect each kept representative pair `(u, v)` with original edge weight (here, `1`).
5. Build an **ALT index** on `G_rep` (CSR + Numba; `L_rep` landmarks).

- Type alias:
```python
ParentALT = Dict[
    Tuple[int, int],  # (parent_level, parent_cid)
    Dict[str, Any]    # { "alt_index": ..., "rep_nodes": {child_cid: Set[int]}, "graph_info": {...} }
]
```
- Function:
```python
def build_parent_alt_indices(
    G: nx.Graph,
    levels: Dict[int, Dict[int, List[int]]],
    pll: Dict[Tuple[int, int], PrunedPLLIndex],
    children_of: Dict[Tuple[int, int], List[int] ],
    node2level2cluster: Dict[int, Dict[int, int]],
    L_rep: int = 32
) -> ParentALT:
    # Build ALT representative-graph indices for every parent cluster that has children.
```

### 5) ALT (CSR + Numba + Bidirectional A*)

Provide these functions (CPU-safe Numba, `int32` distances, `INT32_MAX` for unreachable):
- `nx_to_csr(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]`
- `select_landmarks(G: nx.Graph, L: int = 32, seed: int = 42) -> List[int]` (prefer highest-degree nodes)
- `build_alt_csr(G: nx.Graph, L: int = 32, seed: int = 42) -> Dict[str, Any]` (returns `{ "indptr","indices","lm_dists","mapping","rev_mapping" }`)
- `alt_distance(alt_index: Dict[str, Any], s: int, t: int) -> Tuple[float, Dict[str, int]]` (returns `(distance, {"expanded": int, "relax": int})`, `inf` if unreachable)

### 6) Hierarchical-Hybrid Query

- Function:
```python
def query_hier_hybrid(
    u: int,
    v: int,
    node2level2cluster: Dict[int, Dict[int, int]],
    pll: Dict[Tuple[int, int], PrunedPLLIndex],
    parent_alt: ParentALT,
    max_level: int
) -> Tuple[float, Dict[str, int]]:
    # 1) If u and v are in the same leaf-level cluster (level == max_level):
    #       return PLL[(max_level, cid)].query(u, v).
    # 2) Else, find the smallest level L where cluster(u, L) == cluster(v, L)
    #    (their lowest common parent cluster). Let parent=(L, P). Their
    #    child clusters at level L+1 are Cu and Cv.
    #    For r_u in rep_nodes[Cu], r_v in rep_nodes[Cv]:
    #        d1 = PLL[(L+1, Cu)].query(u, r_u)
    #        d2 = ALT(parent).distance(r_u, r_v)
    #        d3 = PLL[(L+1, Cv)].query(v, r_v)
    #    Take min(d1 + d2 + d3). If no feasible path, return inf.
    #    Return (distance, {"expanded": avg_expanded, "relax": avg_relax}) where applicable; 0 for PLL-only.
```

### 7) Evaluation

- Function:
```python
def evaluate(
    G: nx.Graph,
    seed: int,
    samples: int,
    max_cluster_size: int,
    resolution: float,
    randomness: float,
    L: int,
    L_rep: int
) -> pd.DataFrame:
    # Build and time:
    #   - Whole-graph PLL
    #   - Whole-graph ALT (L landmarks)
    #   - Hier-Hybrid (hierarchical_leiden + per-cluster PLL + parent-level rep-graph ALT)
    # Randomly sample `samples` node pairs (uniform over all nodes). For each pair:
    #   - Compute ground-truth shortest path via NetworkX (weight=1)
    #   - Query all three methods
    # Return DataFrame with exactly columns (and this order):
    # ['method', 'query_time_sec', 'build_time_sec', 'mae']
)
```
- Notes:
  - Use `time.time()` for timing.
  - For each method, accumulate total query time over the sampled pairs; compute MAE over all successful queries (`inf` excluded).
  - Ensure the returned DataFrame is restricted to the four required columns in the specified order.

### 8) Main

- `argparse` options:
  - `--dataset` (default `"PubMed"`; accept `"Pubmed"`)
  - `--seed` (default `42`)
  - `--samples` (default `1000`)
  - `--max-cluster-size` (default `1000`)
  - `--resolution` (default `1.0`)
  - `--randomness` (default `1e-3`)
  - `--L` (default `32`) — landmarks for whole-graph ALT
  - `--L-rep` (default `32`) — landmarks for parent rep-graph ALT
  - `--to-csv` (optional path); if provided, save only the four columns above
- Steps:
  - Set random seeds for `random` and `numpy`.
  - Load graph, run hierarchical detection, build PLL and ALT indices, evaluate.
  - Print: `df[["method","query_time_sec","build_time_sec","mae"]].to_string(index=False)`.
  - If `--to-csv` is set: `df[["method","query_time_sec","build_time_sec","mae"]].to_csv(path, index=False)`.

### 9) Minimal Sanity Tests (non-blocking)

- On a tiny synthetic graph (e.g., grid or star), sample ~20 pairs and assert:
  - Leaf-level same-cluster queries via `query_hier_hybrid` equal NetworkX shortest path.
  - Whole-graph PLL equals NetworkX shortest path.
- If any check fails, print a warning but do not stop the main experiment.

---

## Output Format (exact)

Print a table like this (values are examples only):

```
      method  query_time_sec  build_time_sec   mae
         PLL           0.10           45.80  0.00
         ALT           0.42            3.90  0.00
 Hier-Hybrid           0.06           18.20  0.09
```

No extra columns, no index, and align with the column order: method, query_time_sec, build_time_sec, mae.
