import matplotlib.pyplot as plt
from matplotlib import rcParams
import networkx as nx
import sys
import itertools


# ! I have to copy pairwise from utils.py to avoid circular import


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


isDebug = True if sys.gettrace() else False

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [
    "Consolas",
    "Tahoma",
    "DejaVu Sans",
    "Lucida Grande",
    "Verdana",
]
# rcParams["mathtext.fontset"] = "stix"

def draw_curved_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0.0,
):
    """Draw edge labels.
    Copied from https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)), xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_aggr_graph(G, P, ax=None):
    ax = ax if ax else plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    if pos == {}:
        pos = nx.nx_agraph.graphviz_layout(T, prog="dot")
    T = nx.DiGraph()
    for p in P:
        for node in p:
            T.add_node(node, type=G.nodes[node]["type"])
        for e in itertools.pairwise(p):
            T.add_edge(*e)
    plot(T, pos=pos)


def draw_flow_paths(G, P, ax=None):
    ax = ax if ax else plt.gca()
    pos = nx.get_node_attributes(G, "pos")
    assert pos != {}
    T = nx.DiGraph()
    for i, p in enumerate(P):
        for e in pairwise(p):
            r_e = tuple(reversed(e))
            if T.has_edge(*e):
                T.edges[e]["rad"] += 0.1
            else:
                T.add_edge(e[0], e[1], rad=0)
                if T.has_edge(*r_e):
                    T.edges[e]["rad"] += 0.1
            e_rad = T.edges[e]["rad"]
            nx.draw_networkx_edges(
                T,
                pos,
                arrowstyle="->",
                arrowsize=10,
                width=1,
                edgelist=[e],
                edge_cmap=plt.get_cmap("tab10"),
                edge_vmin=0,
                edge_vmax=len(P),
                edge_color=[i],
                connectionstyle=f"arc3, rad = {e_rad}",
                ax=ax,
            )
    nodes_type = nx.get_node_attributes(G, "type")
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G}, font_size=10, ax=ax)  # resort to node's index
    if nodes_type:
        hosts = [n for n, v in nodes_type.items() if v == "host"]
        switches = [n for n, v in nodes_type.items() if v == "switch"]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=hosts,
            node_shape="s",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=switches,
            node_shape="o",
            node_size=150,
            node_color="xkcd:white",
            edgecolors="k",
            linewidths=1,
            ax=ax,
        )


def plot(
    G: nx.DiGraph | nx.Graph,
    node_label_name=None,
    edge_label_name=None,
    exclude_nodes=[],
    exclude_edges=[],
    pos=None,
    ax=None,
    **kwds,
):
    node_style = dict(node_shape="s", node_size=150, node_color="w", edgecolors="k", linewidths=.5)
    edge_style = dict()
    if "node_shape" in kwds:
        node_style["node_shape"] = kwds["node_shape"]
    if "node_size" in kwds:
        node_style["node_size"] = kwds["node_size"]
    if not ax:
        ax = plt.gca()  # default is gca()

    if isDebug:  # * this is very helpful when you use plot() in debug,
        ax.cla()  # * it will not draw on the same figure
    if not pos:
        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            pos = nx.kamada_kawai_layout(G)
            # G.graph['rankdir'] = 'BT'
            # pos = nx.nx_agraph.graphviz_layout(G, prog='dot') # require pygraphviz
    nodes_type = nx.get_node_attributes(G, "type")

    if nodes_type:  # ! this is for own use to diff hosts and switches
        hosts = [n for n, v in nodes_type.items() if v == "host"]
        switches = [n for n, v in nodes_type.items() if v == "switch"]
        nx.draw_networkx_nodes(G, pos, nodelist=hosts, **node_style, ax=ax)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=switches,
            **node_style,
            ax=ax,
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[n for n in G if n not in exclude_nodes],
            **node_style,
            ax=ax,
        )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=nx.get_node_attributes(G, node_label_name)
        if node_label_name
        else {n: r'${}$'.format(n) for n in G},  # resort to node's index
        font_size=10,
        ax=ax,
        verticalalignment="center_baseline"
    )
    all_edges = list(G.edges())
    edge_labels = nx.get_edge_attributes(G, edge_label_name)

    curved_edges = [e for e in all_edges if tuple(reversed(e)) in all_edges]
    straight_edges = list(set(all_edges) - set(curved_edges))
    if G.is_directed():
        arrow_style = "->"
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle=arrow_style,
            arrowsize=7,
            width=.5,
            edgelist=[e for e in curved_edges if e not in exclude_edges],
            connectionstyle="arc3, rad = 0.1",
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle=arrow_style,
            edgelist=[e for e in straight_edges if e not in exclude_edges],
            arrowsize=7,
            width=.5,
            ax=ax,
        )
    else:
        assert curved_edges == []  # * undirected graph has no curved edges
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[e for e in straight_edges if e not in exclude_edges],
            arrowsize=10,
            width=1,
            ax=ax,
        )

    edge_label_name = edge_label_name if edge_label_name else "weight"

    if edge_labels:
        curved_edge_labels = {edge: edge_labels[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_labels[edge] for edge in straight_edges}
        if curved_edge_labels:
            draw_curved_edge_labels(G, ax=ax, pos=pos, edge_labels=curved_edge_labels, rotate=False, rad=0.1)
        if straight_edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                ax=ax,
                pos=pos,
                bbox={
                    "boxstyle": "square",
                    "ec": "w",
                    "fc": "w",
                    "pad": 0.0,
                },  # edge color and face color
                verticalalignment="bottom",
                edge_labels=straight_edge_labels,
            )

    if isDebug:
        plt.show()


def highlight(
    g: nx.Graph,
    sol: list | nx.DiGraph,
    pos=None,
    node_style=dict(
        node_shape="s",
        node_size=150,
        node_color="yellow",
        edgecolors="red",
        linewidths=1.5,
    ),
    line_style=dict(
        edge_color="r",
        width=1.5,
        arrowstyle="->",
        arrows=True,
        arrowsize=10,
    ),
    ax=None,
) -> None:
    if ax is None:
        ax = plt.gca()
    # for _, attr in g.nodes(data=True):
    #     assert "pos" in attr
    if pos is None:
        try:
            pos = nx.get_node_attributes(g, "pos")
        except AttributeError:
            pass
    if isinstance(sol, list):  # incase a path list
        sol = nx.DiGraph(itertools.pairwise(sol))

    hl_nodes = list(sol.nodes())
    hl_edges = list(sol.edges())
    plot(g, exclude_edges=hl_edges, exclude_nodes=hl_nodes, pos=pos)  # plot original nodes and edges
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=hl_nodes, **node_style)  # highlight the solution nodes
    if sol.is_directed():
        curved_edges = [e for e in sol.edges() if reversed(e) in hl_edges]
        straight_edges = list(set(sol.edges()) - set(curved_edges))
        nx.draw_networkx_edges(
            sol,
            pos,
            ax=ax,
            edgelist=curved_edges,
            **line_style,
            connectionstyle="arc3, rad = 0.1",
        )
        nx.draw_networkx_edges(sol, pos=pos, ax=ax, edgelist=straight_edges, **line_style)
    else:
        del line_style["arrows"]
        del line_style["arrowsize"]
        del line_style["arrowstyle"]
        nx.draw_networkx_edges(sol, pos=pos, ax=ax, edgelist=sol.edges, **line_style)
