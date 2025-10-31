from __future__ import annotations
from typing import List, Tuple
import math
import networkx as nx
import folium

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _heuristic_for(G: nx.Graph):
    def h(u, v):
        yu, xu = float(G.nodes[u]["y"]), float(G.nodes[u]["x"])
        yv, xv = float(G.nodes[v]["y"]), float(G.nodes[v]["x"])
        return haversine_m(yu, xu, yv, xv)
    return h

def _compressed_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """
    Make an undirected, weighted view where each edge (u,v) has weight=min length
    across any parallel directed edges between u and v.
    """
    UG = nx.Graph()
    # copy node coordinates
    for n, d in G.nodes(data=True):
        UG.add_node(n, y=float(d["y"]), x=float(d["x"]))
    for u, v, d in G.edges(data=True):
        w = float(d.get("length", float("inf")))
        if UG.has_edge(u, v):
            if w < float(UG[u][v]["length"]):
                UG[u][v]["length"] = w
        else:
            UG.add_edge(u, v, length=w)
    return UG

def astar_shortest_path(G: nx.MultiDiGraph, source, target) -> List:
    """
    Try directed A* first; if no path, fall back to undirected compressed graph.
    """
    try:
        return nx.astar_path(G, source, target, heuristic=_heuristic_for(G), weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fall back to undirected (ignores one-way direction, but finds connectivity)
        UG = _compressed_undirected(G)
        return nx.astar_path(UG, source, target, heuristic=_heuristic_for(UG), weight="length")

def route_total_length_m(G: nx.MultiDiGraph, route: List) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        # choose shortest parallel edge if present
        min_len = float("inf")
        if G.has_edge(u, v):
            for _, edge in G[u][v].items():
                try:
                    length = float(edge.get("length", float("inf")))
                    if length < min_len:
                        min_len = length
                except Exception:
                    pass
        else:
            # if route came from undirected fallback but (u,v) not present in G directed,
            # add symmetric check
            if G.has_edge(v, u):
                for _, edge in G[v][u].items():
                    try:
                        length = float(edge.get("length", float("inf")))
                        if length < min_len:
                            min_len = length
                    except Exception:
                        pass
        if min_len != float("inf"):
            total += min_len
    return float(total)

def route_coordinates(G: nx.MultiDiGraph, route: List) -> List[Tuple[float, float]]:
    return [(float(G.nodes[n]["y"]), float(G.nodes[n]["x"])) for n in route]

def render_folium_map(coords: List[Tuple[float, float]], a_label: str, b_label: str, out_html_path: str) -> None:
    if not coords:
        raise ValueError("Empty route coordinates.")
    lat_vals = [float(c[0]) for c in coords]
    lon_vals = [float(c[1]) for c in coords]
    center_lat = sum(lat_vals) / float(len(lat_vals))
    center_lon = sum(lon_vals) / float(len(lon_vals))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)
    folium.PolyLine(list(zip(lat_vals, lon_vals)), weight=6, opacity=0.8).add_to(m)
    folium.Marker((lat_vals[0], lon_vals[0]), tooltip="Point A", popup=a_label).add_to(m)
    folium.Marker((lat_vals[-1], lon_vals[-1]), tooltip="Point B", popup=b_label).add_to(m)
    m.save(out_html_path)
