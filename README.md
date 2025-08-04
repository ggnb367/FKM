# FKM

【前置信息】
Pruned Landmark Labeling（PLL）完整流程：
步骤 1：顶点排序
- 对图中的所有顶点 V = {v₁, v₂, ..., vₙ} 进行排序。
- 排序方式可以是按度数、介数中心性等指标降序排列，目的是让重要节点优先处理。

步骤 2：剪枝式标签构建（Pruned BFS/Dijkstra）
- 对于每一个按顺序排列的顶点 vₖ，执行以下操作：
  1. 以 vₖ 为源点执行BFS搜索，逐步访问其他节点 u，计算其距离 δ。
  2. 每当访问一个节点 u，判断以下条件：
     如果已有的标签集合 L′_{k−1} 能够回答：
     Query(vₖ, u, L′_{k−1}) ≤ δ
     → 则对节点 u 进行剪枝：
        - 不将 (vₖ, δ) 添加到 u 的标签中；
        - 不再从 u 出发向外扩展（即不访问 u 的邻居）。
     否则：
        - 将 (vₖ, δ) 添加到 u 的标签中；
        - 继续从 u 向外扩展搜索。
- 对所有顶点重复上述过程，直到所有节点的标签构建完成。

步骤 3：最短路径查询
- 对于任意两个查询节点 s 和 t，执行以下操作：
  1. 分别取出其标签集合 L(s) 和 L(t)；
  2. 找出它们标签中共同的地标节点 l；
  3. 计算并返回：
     
     d(s, t) = min_{l ∈ L(s) ∩ L(t)} [ d(s, l) + d(t, l) ]

- 这是一个典型的 2-hop labeling 查询方式。




【总体思路】
在原始 Pruned Landmark Labeling（PLL）算法基础上，引入Louvain方法对图进行结构划分，并将大图的路径索引任务划分为两个层次：

节点数量大于25的primary Cluster 内部：构建局部 inside-PLL 索引；

节点数量小于25的cluster直接解散，形成零散的点位于primary Cluster 之间：构建抽象化的 outside-PLL 索引，并引入代表点和跳接估值机制。

【具体结构】

**Inside-PLL 构建（含外扩一跳）**

对于primary cluster 中所有边界节点（即连接到 cluster 外部的点），在构建 label 时允许其向外扩展一跳，将相邻的 outside 节点（即“外部边缘节点”）存入一个集合，并和该cluster一起组成一个新的图G_in。

对每个primary cluster及其对应的outside 节点（即“外部边缘节点”）组成的G_in根据每个子图G_in内部的离心度从小到大排序所有节点然后构建inside-PLL。

**Outside Graph 构建**

将每个primary cluster （不包含外部边缘节点）抽象为一个代表节点（代表点）。

把原先的整个大图进行缩小，把所有primary cluster抽象成一个点，然后把所有连接到该cluster 的边都转接到代表点上，形成只包含“代表点 + 零散节点 + 外部边缘点”的新图，称为G_out。

**Outside-PLL 构建**

在G_out上，需要根据介数中心性从高到低进行节点排序然后进行outside-PLL。

在构建过程中，每当“代表点”被访问，使用一个估计的“穿越代价”作为穿过cluster的成本，避免错误地认为 cluster 是一个“1-hop”可通行点。

跳过代表点的穿越代价估计方法如下：

对于 cluster C，定义入边节点集 I_C 和出边节点集 O_C。

计算 I_C 到 O_C 之间的所有最短距离的平均值（向上取整数）作为 cluster C的代表点代价：

w_C = (1 / (|I_C| * |O_C|)) * sum over u in I_C, v in O_C of dist_C(u, v)
在 outside-graph 中，当路径从节点 x 经过代表点 r_C 到达另一个节点 y 时，路径代价为：
dist(x, r_C) + w_C + dist(r_C, y)

【查询过程】

查询任意两个节点 u 和 v 的最短路径估计值：

Step 1：确定 u 所属的 cluster C1 和 v 所属的 cluster C2。

Step 2：在 u 所在 cluster 的 inside-PLL 中找到 u 到最近外部边缘点 e_u 的距离。

Step 3：在 v 所在 cluster 的 inside-PLL 中找到 v 到最近外部边缘点 e_v 的距离。

Step 4：在 outside-PLL 中查询 e_u 到 e_v 的距离（其中可能会经过多个代表点，每个代表点使用预估 w_C 作为跳跃代价）。

Step 5：最终路径长度为：

dist(u, e_u) + dist(e_u, e_v) + dist(e_v, v)
