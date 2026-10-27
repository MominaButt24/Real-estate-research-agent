from pathlib import Path
import json
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def populate_notebook(filename, cells):
    path = NOTEBOOKS / filename

    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    # Preserve the original before replacing it.
    backup = path.with_suffix(
        f".backup-{datetime.now():%Y%m%d-%H%M%S}.ipynb"
    )
    shutil.copy2(path, backup)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    json.loads(temp.read_text(encoding="utf-8"))  # Validate JSON.
    temp.replace(path)

    print(f"Updated: {path.relative_to(ROOT)}")
    print(f"Backup:  {backup.relative_to(ROOT)}")


# ============================================================
# NOTEBOOK 05: LOCAL AND GLOBAL GRAPH SEARCH
# ============================================================

cells05 = [
    markdown("""
# 05 — Local and Global Graph Search

Builds on the Neo4j graph from notebook 04.

- Local retrieval: property, building, and community context.
- Community analysis: counts and price statistics for one community.
- Global retrieval: portfolio-wide statistics and comparisons.

Global aggregation here uses Cypher. It is not the full Microsoft
GraphRAG community-detection and community-report pipeline.
"""),

    markdown("""
## 1. Connect to Neo4j

Expected environment variables: NEO4J_URI, NEO4J_PASSWORD.
NEO4J_USERNAME and NEO4J_DATABASE are optional.
"""),

    code("""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path.cwd()
if not (ROOT / "notebooks").exists() and (ROOT.parent / "notebooks").exists():
    ROOT = ROOT.parent

load_dotenv(ROOT / ".env")
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")
)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise ValueError("Configure NEO4J_URI and NEO4J_PASSWORD in .env")

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)
driver.verify_connectivity()
print("Connected to Neo4j.")


def run_query(cypher, params=None):
    with driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(cypher, params or {})]
"""),

    markdown("""
## 2. Local retrieval

Property IDs are looked up exactly. Community retrieval follows the
existing LOCATED_IN, IN_BUILDING, and HAS_TYPE relationships.
"""),

    code("""
def property_context(property_id):
    return pd.DataFrame(run_query(
        '''
        MATCH (p:Property {property_id: $property_id})
        OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
        OPTIONAL MATCH (p)-[:IN_BUILDING]->(b:Building)
        OPTIONAL MATCH (p)-[:HAS_TYPE]->(t:PropertyType)
        RETURN p.property_id AS property_id,
               p.title AS title,
               p.description AS description,
               p.price AS price,
               p.beds AS beds,
               p.bathrooms AS bathrooms,
               p.area AS area,
               p.listing_type AS listing_type,
               p.status AS status,
               coalesce(c.name, p.community) AS community,
               coalesce(b.name, p.building) AS building,
               coalesce(t.name, p.property_type) AS property_type,
               p.address AS address,
               p.url AS url
        ''',
        {"property_id": str(property_id)}
    ))


def community_properties(community, limit=10):
    return pd.DataFrame(run_query(
        '''
        MATCH (p:Property)-[:LOCATED_IN]->(c:Community)
        WHERE toLower(c.name) CONTAINS toLower($community)
        OPTIONAL MATCH (p)-[:IN_BUILDING]->(b:Building)
        OPTIONAL MATCH (p)-[:HAS_TYPE]->(t:PropertyType)
        RETURN p.property_id AS property_id,
               p.title AS title,
               p.price AS price,
               p.beds AS beds,
               p.bathrooms AS bathrooms,
               p.listing_type AS listing_type,
               p.status AS status,
               c.name AS community,
               coalesce(b.name, p.building) AS building,
               coalesce(t.name, p.property_type) AS property_type,
               p.url AS url
        ORDER BY p.price
        LIMIT $limit
        ''',
        {"community": community, "limit": int(limit)}
    ))


def community_summary(community):
    return pd.DataFrame(run_query(
        '''
        MATCH (p:Property)-[:LOCATED_IN]->(c:Community)
        WHERE toLower(c.name) CONTAINS toLower($community)
        WITH p, c, toFloat(coalesce(p.price, 0)) AS price
        RETURN c.name AS community,
               coalesce(p.listing_type, 'unknown') AS listing_type,
               count(p) AS property_count,
               round(avg(CASE WHEN price > 0 THEN price ELSE NULL END))
                   AS average_price,
               min(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS minimum_price,
               max(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS maximum_price
        ORDER BY property_count DESC
        ''',
        {"community": community}
    ))
"""),

    markdown("""
## 3. Global retrieval

These aggregations deliberately group by listing_type. Sale prices and
rental prices should not be treated as equivalent measurements.
"""),

    code("""
def portfolio_summary():
    return pd.DataFrame(run_query(
        '''
        MATCH (p:Property)
        WITH p, toFloat(coalesce(p.price, 0)) AS price
        RETURN coalesce(p.listing_type, 'unknown') AS listing_type,
               count(p) AS property_count,
               count(DISTINCT p.community) AS communities,
               count(DISTINCT p.building) AS buildings,
               round(avg(CASE WHEN price > 0 THEN price ELSE NULL END))
                   AS average_price,
               min(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS minimum_price,
               max(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS maximum_price
        ORDER BY property_count DESC
        '''
    ))


def compare_communities(limit=20):
    return pd.DataFrame(run_query(
        '''
        MATCH (p:Property)
        WHERE p.community IS NOT NULL
        WITH p, toFloat(coalesce(p.price, 0)) AS price
        RETURN p.community AS community,
               coalesce(p.listing_type, 'unknown') AS listing_type,
               count(p) AS property_count,
               round(avg(CASE WHEN price > 0 THEN price ELSE NULL END))
                   AS average_price,
               min(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS minimum_price,
               max(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS maximum_price
        ORDER BY property_count DESC
        LIMIT $limit
        ''',
        {"limit": int(limit)}
    ))


# Examples: uncomment after connecting.
# display(property_context("4383"))
# display(community_properties("Dubai Marina"))
# display(community_summary("Dubai Marina"))
# display(portfolio_summary())
# display(compare_communities())
"""),

    markdown("""
## 4. Choosing a strategy

| Question | Strategy |
|---|---|
| Tell me about property 4383 | Exact lookup + local graph |
| What is available in Dubai Marina? | Community retrieval, optionally combined with Milvus |
| What is the average price in Dubai Marina? | Community aggregation |
| Compare communities across the portfolio | Global aggregation |

Close the driver when finished.
"""),

    code("""
# Run this cell when you have finished using Neo4j.
# driver.close()
"""),
]


# ============================================================
# NOTEBOOK 06: END-TO-END HYBRID RAG
# ============================================================

cells06 = [
    markdown("""
# 06 — End-to-End Hybrid RAG

Pipeline:
1. Plan the query.
2. Apply explicit hard filters.
3. Retrieve semantic candidates from Milvus.
4. Retrieve keyword matches and connected context from Neo4j.
5. Fuse and deduplicate candidates by property_id.
6. Return evidence for grounded answer generation.

This is a deterministic baseline. The planner is intentionally
replaceable with a structured LLM classifier later.

Existing Milvus collection: property_listings
Embedding model: BAAI/bge-small-en-v1.5
Vector field: embedding
"""),

    markdown("""
## 1. Connect to the existing services

Required environment variables:
MILVUS_URI, NEO4J_URI, NEO4J_PASSWORD.

MILVUS_TOKEN may be omitted for deployments that do not require it.
ANTHROPIC_API_KEY is optional.
"""),

    code("""
import os
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pymilvus import MilvusClient
from neo4j import GraphDatabase

ROOT = Path.cwd()
if not (ROOT / "notebooks").exists() and (ROOT.parent / "notebooks").exists():
    ROOT = ROOT.parent
load_dotenv(ROOT / ".env")
load_dotenv()

COLLECTION_NAME = "property_listings"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

if not os.getenv("MILVUS_URI"):
    raise ValueError("MILVUS_URI is missing from .env")
if not os.getenv("NEO4J_URI") or not os.getenv("NEO4J_PASSWORD"):
    raise ValueError("Configure NEO4J_URI and NEO4J_PASSWORD in .env")

milvus_args = {"uri": os.environ["MILVUS_URI"]}
if os.getenv("MILVUS_TOKEN"):
    milvus_args["token"] = os.environ["MILVUS_TOKEN"]

milvus = MilvusClient(**milvus_args)

if not milvus.has_collection(collection_name=COLLECTION_NAME):
    raise RuntimeError(
        f"Milvus collection {COLLECTION_NAME!r} not found. Run notebook 03 first."
    )
milvus.load_collection(collection_name=COLLECTION_NAME)

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(
        os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")),
        os.environ["NEO4J_PASSWORD"],
    ),
)
neo4j_driver.verify_connectivity()

embedding_model = TextEmbedding(model_name=MODEL_NAME)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print("Connected to Milvus and Neo4j.")


def run_cypher(query, params=None):
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        return [r.data() for r in session.run(query, params or {})]


def embed_query(query):
    text = (
        "Represent this sentence for searching relevant passages: "
        + query.strip()
    )
    vector = list(embedding_model.embed([text]))[0]
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)
"""),

    markdown("""
## 2. Milvus semantic search and filters

The output fields below match notebook 03. Numeric filter fields must
exist in the collection schema. Confirm listing_type labels in your data
before relying on sale/rent filtering.
"""),

    code("""
def escape_milvus_string(value):
    return str(value).replace("\\\\", "\\\\\\\\").replace('"', '\\\\"')


def milvus_search(query, limit=10, filter_expr=None):
    response = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[embed_query(query)],
        anns_field="embedding",
        limit=int(limit),
        filter=filter_expr,
        output_fields=[
            "property_id", "title", "price", "beds", "bathrooms",
            "community", "property_type", "listing_type", "status", "link"
        ],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    records = []
    for hit in (response[0] if response else []):
        e = hit.get("entity", {})
        records.append({
            "property_id": str(e.get("property_id", "")),
            "title": e.get("title"),
            "price": e.get("price"),
            "beds": e.get("beds"),
            "bathrooms": e.get("bathrooms"),
            "community": e.get("community"),
            "property_type": e.get("property_type"),
            "listing_type": e.get("listing_type"),
            "status": e.get("status"),
            "link": e.get("link"),
            "vector_score": hit.get("distance"),
            "source": "milvus",
        })
    return pd.DataFrame(records)
"""),

    markdown("""
## 3. Neo4j local and global retrieval

These functions reuse the existing graph schema:
Property, Community, Building, PropertyType, and the relationships
LOCATED_IN, IN_BUILDING, and HAS_TYPE.
"""),

    code("""
def graph_properties_by_ids(property_ids):
    ids = [str(x) for x in property_ids if str(x).strip()]
    if not ids:
        return pd.DataFrame()

    return pd.DataFrame(run_cypher(
        '''
        MATCH (p:Property)
        WHERE toString(p.property_id) IN $ids
        OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
        OPTIONAL MATCH (p)-[:IN_BUILDING]->(b:Building)
        OPTIONAL MATCH (p)-[:HAS_TYPE]->(t:PropertyType)
        RETURN p.property_id AS property_id,
               p.title AS title,
               p.description AS description,
               p.price AS price,
               p.beds AS beds,
               p.bathrooms AS bathrooms,
               p.area AS area,
               p.listing_type AS listing_type,
               p.status AS status,
               coalesce(c.name, p.community) AS community,
               coalesce(b.name, p.building) AS building,
               coalesce(t.name, p.property_type) AS property_type,
               p.address AS address,
               p.url AS url
        ''',
        {"ids": ids}
    ))


def graph_keyword_search(query, limit=10):
    terms = [
        t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query)
        if len(t) > 2
    ]
    if not terms:
        return pd.DataFrame()

    return pd.DataFrame(run_cypher(
        '''
        MATCH (p:Property)
        WITH p,
             toLower(coalesce(p.title, '') + ' ' +
                     coalesce(p.description, '') + ' ' +
                     coalesce(p.community, '') + ' ' +
                     coalesce(p.building, '') + ' ' +
                     coalesce(p.property_type, '') + ' ' +
                     coalesce(p.address, '')) AS searchable
        WITH p, size([
            term IN $terms WHERE searchable CONTAINS term
        ]) AS lexical_score
        WHERE lexical_score > 0
        RETURN p.property_id AS property_id,
               p.title AS title,
               p.price AS price,
               p.beds AS beds,
               p.bathrooms AS bathrooms,
               p.community AS community,
               p.property_type AS property_type,
               p.listing_type AS listing_type,
               p.status AS status,
               p.url AS url,
               lexical_score
        ORDER BY lexical_score DESC
        LIMIT $limit
        ''',
        {"terms": terms, "limit": int(limit)}
    ))


def graph_portfolio_summary():
    return pd.DataFrame(run_cypher(
        '''
        MATCH (p:Property)
        WITH p, toFloat(coalesce(p.price, 0)) AS price
        RETURN coalesce(p.listing_type, 'unknown') AS listing_type,
               count(p) AS property_count,
               count(DISTINCT p.community) AS communities,
               count(DISTINCT p.building) AS buildings,
               round(avg(CASE WHEN price > 0 THEN price ELSE NULL END))
                   AS average_price,
               min(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS minimum_price,
               max(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS maximum_price
        ORDER BY property_count DESC
        '''
    ))


def graph_community_summary(community):
    return pd.DataFrame(run_cypher(
        '''
        MATCH (p:Property)-[:LOCATED_IN]->(c:Community)
        WHERE toLower(c.name) CONTAINS toLower($community)
        WITH p, c, toFloat(coalesce(p.price, 0)) AS price
        RETURN c.name AS community,
               coalesce(p.listing_type, 'unknown') AS listing_type,
               count(p) AS property_count,
               round(avg(CASE WHEN price > 0 THEN price ELSE NULL END))
                   AS average_price,
               min(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS minimum_price,
               max(CASE WHEN price > 0 THEN price ELSE NULL END)
                   AS maximum_price
        ORDER BY property_count DESC
        ''',
        {"community": community}
    ))
"""),

    markdown("""
## 4. Query planner

This starter planner extracts a few common explicit constraints. It is
not a full natural-language understanding system. Replace or extend it
after testing against real user queries.
"""),

    code("""
# IMPORTANT: update these lists to match actual values in your Milvus data.
SALE_VALUES = ["sale", "purchase", "buy"]
RENT_VALUES = ["rent", "rental", "lease"]


def plan_query(question):
    q = question.strip()
    low = q.lower()

    plan = {
        "question": q,
        "intent": "property_search",
        "scope": "local",
        "property_id": None,
        "community": None,
        "filters": {},
    }

    m = re.search(
        r"\\b(?:property|listing|id)\\s*#?\\s*(\\d{3,})\\b", q, re.I
    )
    if m:
        plan["intent"] = "property_lookup"
        plan["property_id"] = m.group(1)
        return plan

    m = re.search(
        r"\\b(?:under|below|up to|less than|max(?:imum)?(?: budget)?(?: of)?)"
        r"\\s*(?:aed\\s*)?([\\d,]+)",
        low,
    )
    if m:
        plan["filters"]["price_max"] = float(m.group(1).replace(",", ""))

    m = re.search(
        r"\\b(?:above|over|more than|at least)\\s*(?:aed\\s*)?([\\d,]+)",
        low,
    )
    if m:
        plan["filters"]["price_min"] = float(m.group(1).replace(",", ""))

    m = re.search(r"\\b(\\d+)\\s*(?:bedrooms?|beds?|br)\\b", low)
    if m:
        plan["filters"]["beds"] = int(m.group(1))

    m = re.search(r"\\b(\\d+)\\s*bathrooms?\\b", low)
    if m:
        plan["filters"]["bathrooms_min"] = int(m.group(1))

    # Prefer a community dictionary for reliable extraction.
    # This list is only a starter example; add your dataset's communities.
    known_communities = [
        "Dubai Marina", "Downtown Dubai", "Business Bay",
        "Arabian Ranches", "Palm Jumeirah", "Jumeirah Village Circle",
    ]
    for name in sorted(known_communities, key=len, reverse=True):
        if name.lower() in low:
            plan["community"] = name
            break

    if re.search(r"\\b(rent|rental|lease|to rent)\\b", low):
        plan["filters"]["transaction_type"] = "rent"
    elif re.search(r"\\b(sale|for sale|purchase|buy|buying)\\b", low):
        plan["filters"]["transaction_type"] = "sale"

    global_terms = [
        "portfolio", "overall", "across all", "market trends",
        "compare communities", "global overview", "inventory by community",
    ]
    if any(term in low for term in global_terms):
        plan["intent"] = "portfolio_analysis"
        plan["scope"] = "global"
    elif (
        "average price in" in low
        or "how many in" in low
        or "inventory in" in low
    ):
        plan["intent"] = "community_analysis"

    return plan


def build_milvus_filter(plan):
    f = plan["filters"]
    parts = []

    if "price_min" in f:
        parts.extend([
            "price > 0",
            f"price >= {float(f['price_min'])}",
        ])
    if "price_max" in f:
        parts.extend([
            "price > 0",
            f"price <= {float(f['price_max'])}",
        ])
    if "beds" in f:
        parts.append(f"beds == {int(f['beds'])}")
    if "bathrooms_min" in f:
        parts.append(f"bathrooms >= {float(f['bathrooms_min'])}")
    if plan["community"]:
        community = escape_milvus_string(plan["community"])
        parts.append(f'community like "%{community}%"')

    transaction = f.get("transaction_type")
    values = SALE_VALUES if transaction == "sale" else RENT_VALUES if transaction == "rent" else None
    if values:
        quoted = ", ".join(
            '"' + escape_milvus_string(v) + '"' for v in values
        )
        parts.append(f"listing_type in [{quoted}]")

    return " and ".join(parts) or None


for example in [
    "Find 2 bedrooms under AED 2,000,000 in Dubai Marina for sale",
    "Compare communities across the portfolio",
    "Tell me about property 4383",
]:
    print(example)
    print(json.dumps(plan_query(example), indent=2))
"""),

    markdown("""
## 5. Candidate fusion and retrieval

This baseline uses vector similarity plus keyword score. Milvus COSINE
scores and lexical scores are not necessarily calibrated on the same
scale, so treat this as a starting point and evaluate it on labelled
queries before production use.
"""),

    code("""
def fuse_candidates(vector_df, lexical_df, limit=10):
    frames = [
        df for df in (vector_df, lexical_df)
        if df is not None and not df.empty
    ]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "property_id" not in df:
        return df.head(limit)

    df["property_id"] = df["property_id"].astype(str)
    df = df[df["property_id"].str.strip().ne("")].copy()

    df["vector_score"] = pd.to_numeric(
        df.get("vector_score", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    ).fillna(0)
    df["lexical_score"] = pd.to_numeric(
        df.get("lexical_score", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    ).fillna(0)

    vmax = df["vector_score"].max()
    lmax = df["lexical_score"].max()

    df["vector_norm"] = df["vector_score"] / vmax if vmax > 0 else 0
    df["lexical_norm"] = df["lexical_score"] / lmax if lmax > 0 else 0
    df["fusion_score"] = (
        0.7 * df["vector_norm"] + 0.3 * df["lexical_norm"]
    )

    df = df.sort_values("fusion_score", ascending=False)
    df = df.groupby("property_id", as_index=False, sort=False).first()
    return df.sort_values(
        "fusion_score", ascending=False
    ).head(limit).reset_index(drop=True)


def exact_property_lookup(property_id):
    graph_df = graph_properties_by_ids([property_id])
    if not graph_df.empty:
        return graph_df

    rows = milvus.get(
        collection_name=COLLECTION_NAME,
        ids=[str(property_id)],
        output_fields=[
            "property_id", "title", "price", "beds", "bathrooms",
            "community", "property_type", "listing_type", "status",
            "link", "search_text",
        ],
    )
    return pd.DataFrame(rows)


def retrieve(question, limit=10):
    plan = plan_query(question)

    if plan["intent"] == "property_lookup":
        return {
            "plan": plan,
            "mode": "exact_lookup",
            "results": exact_property_lookup(plan["property_id"]),
        }

    if plan["intent"] == "portfolio_analysis":
        return {
            "plan": plan,
            "mode": "global_aggregation",
            "results": graph_portfolio_summary(),
        }

    if plan["intent"] == "community_analysis" and plan["community"]:
        return {
            "plan": plan,
            "mode": "community_aggregation",
            "results": graph_community_summary(plan["community"]),
        }

    filter_expr = build_milvus_filter(plan)
    vector_df = milvus_search(
        question, limit=limit * 2, filter_expr=filter_expr
    )
    lexical_df = graph_keyword_search(question, limit=limit * 2)

    # Restrict keyword results to the requested community, when recognized.
    if plan["community"] and not lexical_df.empty:
        lexical_df = lexical_df[
            lexical_df["community"].fillna("").astype(str).str.contains(
                plan["community"], case=False, regex=False
            )
        ]

    candidates = fuse_candidates(vector_df, lexical_df, limit=limit)
    ids = candidates["property_id"].tolist() if not candidates.empty else []
    graph_df = graph_properties_by_ids(ids)

    if not graph_df.empty:
        scores = candidates.drop_duplicates("property_id")
        keep = [
            "property_id", "vector_score", "lexical_score", "fusion_score"
        ]
        scores = scores[[c for c in keep if c in scores.columns]]
        result = graph_df.merge(scores, on="property_id", how="left")
    else:
        result = candidates

    # Final numeric checks help protect against filter/data inconsistencies.
    f = plan["filters"]
    if not result.empty and "price" in result:
        prices = pd.to_numeric(result["price"], errors="coerce")
        if "price_min" in f:
            result = result[prices >= f["price_min"]]
            prices = pd.to_numeric(result["price"], errors="coerce")
        if "price_max" in f:
            result = result[prices <= f["price_max"]]

    if not result.empty and "beds" in result and "beds" in f:
        result = result[
            pd.to_numeric(result["beds"], errors="coerce") == f["beds"]
        ]

    if (
        not result.empty
        and "bathrooms" in result
        and "bathrooms_min" in f
    ):
        result = result[
            pd.to_numeric(result["bathrooms"], errors="coerce")
            >= f["bathrooms_min"]
        ]

    if plan["community"] and "community" in result:
        result = result[
            result["community"].fillna("").astype(str).str.contains(
                plan["community"], case=False, regex=False
            )
        ]

    return {
        "plan": plan,
        "mode": "hybrid_local_search",
        "results": result.reset_index(drop=True),
    }
"""),

    markdown("""
## 6. Grounded answer generation

Without ANTHROPIC_API_KEY, this returns retrieved evidence directly.
When configured, Anthropic receives the evidence and instructions not
to invent listing details or statistics.
"""),

    code("""
def answer_question(question):
    payload = retrieve(question)
    print("PLAN:")
    print(json.dumps(payload["plan"], indent=2, default=str))
    print("\\nRETRIEVAL MODE:", payload["mode"])

    results = payload["results"]
    if results.empty:
        print("No matching evidence was returned. Try broadening the query.")
        return payload

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            evidence = results.head(10).to_json(
                orient="records", indent=2, force_ascii=False
            )
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                max_tokens=900,
                system=(
                    "You are a real-estate research assistant. Answer only "
                    "from the supplied evidence. Never invent prices, "
                    "availability, property details, or trends. State when "
                    "evidence is insufficient and mention listing data may "
                    "be stale."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        "Question: " + question + "\\nEvidence:\\n" + evidence
                    ),
                }],
            )
            print("\\nANSWER:")
            print("\\n".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            ))
        except Exception as exc:
            print("LLM generation failed; showing retrieved evidence:", exc)
            display(results)
    else:
        print(
            "\\nANTHROPIC_API_KEY not configured. "
            "Showing evidence without generated prose."
        )
        display(results)

    return payload


# Try these after checking connectivity and dataset values:
# answer_question("Find 2 bedrooms under AED 2,000,000 in Dubai Marina")
# answer_question("Compare communities across the portfolio")
# answer_question("What is the average price in Dubai Marina?")
# answer_question("Tell me about property 4383")
"""),

    markdown("""
## 7. Planner smoke tests and limitations

These tests check planning logic, not live database correctness or
answer quality. Verify listing_type labels and community names against
the real dataset. Add a labelled query set to evaluate Recall@k,
filter correctness, and answer faithfulness.
"""),

    code("""
def test_planner():
    p = plan_query("Find 2 bedrooms under AED 2,000,000")
    assert p["filters"]["beds"] == 2
    assert p["filters"]["price_max"] == 2_000_000

    p = plan_query("Compare communities across the portfolio")
    assert p["intent"] == "portfolio_analysis"

    p = plan_query("Tell me about property 4383")
    assert p["property_id"] == "4383"

    print("Planner smoke tests passed.")


test_planner()

# Close clients when finished:
# neo4j_driver.close()
# milvus.close()
"""),
]


def main():
    if not NOTEBOOKS.is_dir():
        raise FileNotFoundError(
            f"Expected a notebooks/ folder inside repository root: {ROOT}"
        )

    populate_notebook("05_graph_local_global_search.ipynb", cells05)
    populate_notebook("06_hybrid_rag_end_to_end.ipynb", cells06)

    print("\\nDone. Both notebooks were populated.")
    print("Open them in Jupyter or VS Code and run cells in order.")
    print("Database-dependent cells require your .env credentials.")


if __name__ == "__main__":
    main()