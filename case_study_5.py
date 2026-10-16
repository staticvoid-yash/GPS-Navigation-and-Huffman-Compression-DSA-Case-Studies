"""
============================================================
 Case Study: GPS / Navigation System
 City map modeled as a Weighted Graph
 Algorithms: Dijkstra, A*, Bellman-Ford, Floyd-Warshall
============================================================
"""

import heapq
import math
from collections import defaultdict


class CityMap:
    def __init__(self):
        # adjacency list: node -> [(neighbor, distance_km, speed_kmph)]
        self.graph = defaultdict(list)
        self.coords = {}       # node -> (lat, lon) for A* heuristic

    # ---------- BUILD MAP ----------
    def add_location(self, name, lat=0.0, lon=0.0):
        self.coords[name] = (lat, lon)
        if name not in self.graph:
            self.graph[name] = []

    def add_road(self, u, v, distance_km, speed_kmph=40, bidirectional=True):
        """Weight = time in minutes (distance/speed)."""
        time_min = (distance_km / speed_kmph) * 60
        self.graph[u].append((v, distance_km, time_min))
        if bidirectional:
            self.graph[v].append((u, distance_km, time_min))

    # ---------- 1. DIJKSTRA (Shortest Path) ----------
    def dijkstra(self, src, dst, weight="time"):
        """
        weight = 'distance' → shortest km
        weight = 'time'     → fastest route (default)
        """
        idx = 1 if weight == "distance" else 2
        dist = {node: float("inf") for node in self.graph}
        prev = {node: None for node in self.graph}
        dist[src] = 0
        pq = [(0, src)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break

            for v, km, t in self.graph[u]:
                w = km if weight == "distance" else t
                if d + w < dist[v]:
                    dist[v] = d + w
                    prev[v] = u
                    heapq.heappush(pq, (dist[v], v))

        path = self._reconstruct(prev, src, dst)
        return path, dist[dst]

    def _reconstruct(self, prev, src, dst):
        if prev[dst] is None and src != dst:
            return None
        path, cur = [], dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    # ---------- 2. A* SEARCH (Faster) ----------
    def a_star(self, src, dst, weight="time"):
        """A* with Euclidean heuristic based on coordinates."""
        idx = 1 if weight == "distance" else 2

        def heuristic(a, b):
            (lat1, lon1), (lat2, lon2) = self.coords[a], self.coords[b]
            return math.hypot(lat1 - lat2, lon1 - lon2) * 111  # ~km

        open_set = [(0, src)]
        g_score = {node: float("inf") for node in self.graph}
        g_score[src] = 0
        prev = {node: None for node in self.graph}
        visited = set()

        while open_set:
            _, u = heapq.heappop(open_set)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break

            for v, km, t in self.graph[u]:
                w = km if weight == "distance" else t
                tentative = g_score[u] + w
                if tentative < g_score[v]:
                    g_score[v] = tentative
                    prev[v] = u
                    f = tentative + heuristic(v, dst)
                    heapq.heappush(open_set, (f, v))

        path = self._reconstruct(prev, src, dst)
        return path, g_score[dst]

    # ---------- 3. BELLMAN-FORD (Negative Weights) ----------
    def bellman_ford(self, src):
        """Handles negative weights (e.g., toll discounts)."""
        dist = {node: float("inf") for node in self.graph}
        dist[src] = 0

        for _ in range(len(self.graph) - 1):
            for u in self.graph:
                for v, km, t in self.graph[u]:
                    if dist[u] + t < dist[v]:
                        dist[v] = dist[u] + t

        # Detect negative cycle
        for u in self.graph:
            for v, km, t in self.graph[u]:
                if dist[u] + t < dist[v]:
                    print("  ⚠ Negative cycle detected!")
                    return None
        return dist

    # ---------- 4. FLOYD-WARSHALL (All-Pairs) ----------
    def floyd_warshall(self):
        nodes = list(self.graph.keys())
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        INF = float("inf")

        D = [[INF] * n for _ in range(n)]
        for i in range(n):
            D[i][i] = 0
        for u in self.graph:
            for v, km, t in self.graph[u]:
                D[idx[u]][idx[v]] = min(D[idx[u]][idx[v]], t)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if D[i][k] + D[k][j] < D[i][j]:
                        D[i][j] = D[i][k] + D[k][j]
        return nodes, D

    # ---------- 5. K-NEAREST LANDMARKS ----------
    def nearest_landmarks(self, src, k=3):
        """Find k closest locations by shortest time."""
        dist = {node: float("inf") for node in self.graph}
        dist[src] = 0
        pq = [(0, src)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            for v, km, t in self.graph[u]:
                if d + t < dist[v]:
                    dist[v] = d + t
                    heapq.heappush(pq, (dist[v], v))

        nearest = sorted(dist.items(), key=lambda x: x[1])
        return [(n, round(d, 1)) for n, d in nearest if n != src][:k]

    # ---------- 6. DISPLAY ----------
    def show(self):
        print("\n  ── City Map (adjacency list) ──")
        for u, edges in self.graph.items():
            conn = ", ".join(f"{v}({km}km,{t:.0f}min)" for v, km, t in edges)
            print(f"  {u:<12} → {conn}")


# ============================================================
# DEMO
# ============================================================
def gps_demo():
    print("=" * 70)
    print("   CASE STUDY : GPS / NAVIGATION SYSTEM (Weighted Graph)")
    print("=" * 70)

    city = CityMap()

    # Locations (name, lat, lon)
    locs = [
        ("Home",       12.9716, 77.5946),
        ("Office",     12.9352, 77.6245),
        ("Mall",       12.9592, 77.6974),
        ("Airport",    13.1986, 77.7066),
        ("Station",    12.9784, 77.5726),
        ("Hospital",   12.9698, 77.7500),
        ("University", 12.9345, 77.6050),
        ("Stadium",    12.9788, 77.5996),
    ]
    for name, lat, lon in locs:
        city.add_location(name, lat, lon)

    # Roads (u, v, km, speed kmph)
    roads = [
        ("Home",       "Office",     8,  30),
        ("Home",       "Station",    3,  25),
        ("Office",     "University", 4,  20),
        ("Office",     "Mall",       10, 40),
        ("Mall",       "Hospital",   6,  30),
        ("Mall",       "Airport",    35, 60),
        ("Station",    "Stadium",    5,  25),
        ("Stadium",    "Office",     7,  30),
        ("University", "Hospital",   12, 35),
        ("Hospital",   "Airport",    40, 70),
        ("Station",    "University", 9,  28),
    ]
    for r in roads:
        city.add_road(*r)

    city.show()

    # 1. Dijkstra — Fastest route
    print("\n  ── 1. Dijkstra (Fastest Route: Home → Airport) ──")
    path, t = city.dijkstra("Home", "Airport", weight="time")
    print(f"  Path  : {' → '.join(path)}")
    print(f"  Time  : {t:.1f} minutes")

    # 2. Dijkstra — Shortest distance
    print("\n  ── 2. Dijkstra (Shortest Distance: Home → Airport) ──")
    path, d = city.dijkstra("Home", "Airport", weight="distance")
    print(f"  Path  : {' → '.join(path)}")
    print(f"  Dist  : {d:.1f} km")

    # 3. A* Search
    print("\n  ── 3. A* Search (Home → Airport) ──")
    path, t = city.a_star("Home", "Airport", weight="time")
    print(f"  Path  : {' → '.join(path)}")
    print(f"  Time  : {t:.1f} minutes")

    # 4. Bellman-Ford
    print("\n  ── 4. Bellman-Ford (Single Source: Home) ──")
    dist = city.bellman_ford("Home")
    for node, t in dist.items():
        print(f"    Home → {node:<12} : {t:.1f} min")

    # 5. Floyd-Warshall
    print("\n  ── 5. Floyd-Warshall (All-Pairs Shortest Time) ──")
    nodes, D = city.floyd_warshall()
    print("  Matrix (minutes) — top 5 rows:")
    print("         " + " ".join(f"{n[:4]:>6}" for n in nodes[:5]))
    for i, row in enumerate(D[:5]):
        print(f"  {nodes[i][:5]:<6} " + " ".join(f"{v:>6.0f}" for v in row[:5]))

    # 6. K-Nearest
    print("\n  ── 6. Nearest Landmarks from Home ──")
    for name, t in city.nearest_landmarks("Home", k=3):
        print(f"    {name:<12} : {t} min")


if __name__ == "__main__":
    gps_demo()