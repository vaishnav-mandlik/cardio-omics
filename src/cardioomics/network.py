from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import requests

EMPTY = ["gene_a", "gene_b", "score"]


def fetch_string_network(
    genes: list[str],
    cache_dir: str | Path,
    *,
    api: str = "https://version-12-0.string-db.org/api",
    required_score: int = 700,
    species: int = 9606,
    offline: bool = False,
) -> pd.DataFrame:
    genes = sorted(set(genes))
    if not genes:
        return pd.DataFrame(columns=EMPTY)
    key = hashlib.sha256(f"{species}|{required_score}|{','.join(genes)}".encode()).hexdigest()[:20]
    cache = Path(cache_dir) / f"string_{key}.json"
    if cache.exists():
        edges = json.loads(cache.read_text())
    elif offline:
        raise FileNotFoundError(f"offline mode and no cached STRING response at {cache}")
    else:
        resp = requests.post(
            f"{api}/json/network",
            data={
                "identifiers": "\r".join(genes),
                "species": species,
                "required_score": required_score,
                "caller_identity": "cardioomics",
            },
            timeout=120,
        )
        resp.raise_for_status()
        edges = resp.json()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(edges))
    df = pd.DataFrame(edges)
    if df.empty:
        return pd.DataFrame(columns=EMPTY)
    df = df.rename(columns={"preferredName_A": "gene_a", "preferredName_B": "gene_b"})
    return df[EMPTY].drop_duplicates()


def hub_table(edges: pd.DataFrame, node_attrs: pd.DataFrame | None = None) -> tuple[nx.Graph, pd.DataFrame]:
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(row.gene_a, row.gene_b, weight=float(row.score))
    if G.number_of_nodes() == 0:
        return G, pd.DataFrame(columns=["gene", "degree", "betweenness"])
    btw = nx.betweenness_centrality(G, normalized=True, seed=0)
    hubs = pd.DataFrame(
        {"gene": list(G.nodes), "degree": [G.degree(n) for n in G.nodes], "betweenness": [btw[n] for n in G.nodes]}
    ).sort_values(["degree", "betweenness"], ascending=False)
    if node_attrs is not None:
        hubs = hubs.merge(node_attrs, left_on="gene", right_index=True, how="left")
    return G, hubs.reset_index(drop=True)
