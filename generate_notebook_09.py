
#!/usr/bin/env python3
"""Generate Notebook 09: Hybrid RAG Integration Pipeline.

Usage:
    python generate_notebook_09.py

Optional:
    python generate_notebook_09.py --output notebooks/09_hybrid_rag_integration.ipynb
"""

import argparse
import json
from pathlib import Path


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [

    # 1. Environment and connections
    markdown("""## 1. Environment and connections
Load environment variables and initialize Anthropic, Milvus, Neo4j, and embeddings."""),
    code(r'''
import os
import re
import json
import math
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from anthropic import Anthropic
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from fastembed import TextEmbedding

ROOT = Path.cwd()
if not (ROOT / "notebooks").exists() and (ROOT.parent / "notebooks").exists():
    ROOT = ROOT.parent

load_dotenv(ROOT / ".env")
load_dotenv()

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "property_listings")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

INTENT_CATEGORIES = [
    "Family-Friendly",
    "Investment",
    "Lifestyle",
    "Views",
]

required_env = [
    "MILVUS_URI",
    "NEO4J_URI",
    "NEO4J_PASSWORD",
    "ANTHROPIC_API_KEY",
]

missing = [key for key in required_env if not os.getenv(key)]
if missing:
    raise EnvironmentError(
        f"Missing environment variables: {missing}"
    )

milvus_args = {"uri": os.environ["MILVUS_URI"]}
if os.getenv("MILVUS_TOKEN"):
    milvus_args["token"] = os.environ["MILVUS_TOKEN"]

milvus = MilvusClient(**milvus_args)

if not milvus.has_collection(COLLECTION_NAME):
    raise RuntimeError(
        f"Milvus collection not found: {COLLECTION_NAME}"
    )

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(
        os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")),
        os.environ["NEO4J_PASSWORD"],
    ),
)
neo4j_driver.verify_connectivity()

anthropic_client = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

print("Connections initialized.")
print("Milvus:", COLLECTION_NAME)
print("Anthropic:", ANTHROPIC_MODEL)


def run_cypher(query, parameters=None):
    with neo4j_driver.session(
        database=NEO4J_DATABASE
    ) as session:
        return [
            record.data()
            for record in session.run(query, parameters or {})
        ]
'''),

    # 2. Customer profile
    markdown("""## 2. Customer profile and validation
Define persistent customer preferences, validate proposed updates, and merge them safely."""),
    code(r'''
PROFILE_TEMPLATE = {
    "transaction_type": None,
    "property_type": [],
    "hard_constraints": {
        "budget_min": None,
        "budget_max": None,
        "bedrooms_min": None,
        "bedrooms_max": None,
        "bathrooms_min": None,
        "communities": [],
        "required_features": [],
        "excluded_features": [],
        "other": [],
    },
    "intent_preferences": {
        name: 0.0 for name in INTENT_CATEGORIES
    },
    "soft_preferences": {
        "preferred_communities": [],
        "features": [],
        "notes": [],
    },
}

PROFILE_STORE = {}
HISTORY_STORE = {}


def new_profile():
    return json.loads(json.dumps(PROFILE_TEMPLATE))


def clean_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []

    result = []
    for item in value:
        item = str(item).strip()
        if item and item not in result:
            result.append(item)
    return result


def as_number(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return None
        return int(value) if value.is_integer() else value
    except (TypeError, ValueError):
        return None


def validate_profile_patch(patch):
    """Allow only recognized profile fields and normalize their values."""
    if not isinstance(patch, dict):
        return {}

    safe = {}

    if patch.get("transaction_type") in ("sale", "rent"):
        safe["transaction_type"] = patch["transaction_type"]

    if "property_type" in patch:
        safe["property_type"] = clean_list(patch["property_type"])

    hard = patch.get("hard_constraints")
    if isinstance(hard, dict):
        safe_hard = {}

        numeric_fields = [
            "budget_min", "budget_max",
            "bedrooms_min", "bedrooms_max",
            "bathrooms_min",
        ]

        list_fields = [
            "communities", "required_features",
            "excluded_features", "other",
        ]

        for key in numeric_fields:
            if key in hard:
                safe_hard[key] = as_number(hard[key])

        for key in list_fields:
            if key in hard:
                safe_hard[key] = clean_list(hard[key])

        if safe_hard:
            safe["hard_constraints"] = safe_hard

    intents = patch.get("intent_preferences")
    if isinstance(intents, dict):
        safe_intents = {}

        for name, value in intents.items():
            if name not in INTENT_CATEGORIES:
                continue
            try:
                score = float(value)
                if math.isfinite(score):
                    safe_intents[name] = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                continue

        if safe_intents:
            safe["intent_preferences"] = safe_intents

    soft = patch.get("soft_preferences")
    if isinstance(soft, dict):
        safe_soft = {}

        for key in ("preferred_communities", "features", "notes"):
            if key in soft:
                safe_soft[key] = clean_list(soft[key])

        if safe_soft:
            safe["soft_preferences"] = safe_soft

    return safe


def merge_dict(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(target[key], value)
        elif isinstance(value, list):
            target[key] = clean_list(value)
        else:
            target[key] = value

    return target


def merge_profile(profile, patch):
    profile = json.loads(json.dumps(profile))
    patch = validate_profile_patch(patch)
    return merge_dict(profile, patch)
'''),

    # 3. Query planning and profile extraction
    markdown("""## 3. Query planning and profile extraction
Use one LLM call to classify the request and propose customer-profile updates; validate its JSON in Python."""),
    code(r'''
CLASSIFIER_PROMPT = """
You are the query planner for a Dubai real-estate research assistant.

Return ONLY a valid JSON object with these keys:
query_type, normalized_query, transaction_type, property_types,
hard_constraints, soft_preferences, intent_preferences,
exact_property_id, community, answer_mode, profile_updates,
clarification_needed, clarification_question.

query_type must be one of:
exact_lookup, property_search, community_analysis,
portfolio_analysis, general_real_estate.

answer_mode must be grounded_answer or data_summary.

Use null for unknown scalar values and [] for unknown lists.
Do not invent preferences.
Separate hard constraints from soft preferences.
A preferred community is not a hard constraint unless the user
explicitly says they will only consider that community.
transaction_type must be sale, rent, or null.
property_types must be a list.
intent_preferences may contain only these categories:
Family-Friendly, Investment, Lifestyle, Views.
Intent scores must be between 0 and 1.

profile_updates must use the customer profile schema.
Only save durable preferences to the profile, not temporary
requirements specific to the current search.
exact_property_id must be null unless a property ID is explicit.
"""


def parse_json_object(text):
    text = (text or "").strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.I,
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM did not return a JSON object.")
        data = json.loads(text[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")

    return data


def classify_query(query, profile, history=None):
    payload = {
        "query": query,
        "current_profile": profile,
        "recent_history": (history or [])[-6:],
        "valid_intents": INTENT_CATEGORIES,
    }

    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1400,
        temperature=0,
        system=CLASSIFIER_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False
                ),
            }
        ],
    )

    raw = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )

    plan = parse_json_object(raw)

    allowed_types = {
        "exact_lookup",
        "property_search",
        "community_analysis",
        "portfolio_analysis",
        "general_real_estate",
    }

    if plan.get("query_type") not in allowed_types:
        plan["query_type"] = "property_search"

    if plan.get("transaction_type") not in ("sale", "rent"):
        plan["transaction_type"] = None

    plan["normalized_query"] = str(
        plan.get("normalized_query") or query
    ).strip()

    plan["property_types"] = clean_list(
        plan.get("property_types")
    )

    plan["hard_constraints"] = (
        plan.get("hard_constraints")
        if isinstance(plan.get("hard_constraints"), dict)
        else {}
    )

    plan["soft_preferences"] = (
        plan.get("soft_preferences")
        if isinstance(plan.get("soft_preferences"), dict)
        else {}
    )

    plan["intent_preferences"] = (
        plan.get("intent_preferences")
        if isinstance(plan.get("intent_preferences"), dict)
        else {}
    )

    plan["profile_updates"] = validate_profile_patch(
        plan.get("profile_updates", {})
    )

    return plan
'''),

    # 4. Hybrid retrieval
    markdown("""## 4. Hybrid retrieval
Search Milvus with dense embeddings and Neo4j with lexical matching; fuse both result lists using reciprocal-rank fusion."""),
    code(r'''
def embed_query(query):
    text = (
        "Represent this sentence for searching relevant passages: "
        + query
    )
    return list(next(iter(embedding_model.query_embed(text))))


def milvus_search(query, limit=30):
    results = milvus.search(
        collection_name=COLLECTION_NAME,
        data=[embed_query(query)],
        anns_field="embedding",
        limit=limit,
        output_fields=[
            "property_id", "title", "price", "beds",
            "bathrooms", "community", "property_type",
            "listing_type", "status", "link",
        ],
        search_params={
            "metric_type": "COSINE",
            "params": {},
        },
    )

    rows = []
    for hit in (results[0] if results else []):
        row = dict(hit.get("entity") or {})
        row["property_id"] = str(
            row.get("property_id", hit.get("id", ""))
        )
        row["vector_score"] = float(
            hit.get("distance", hit.get("score", 0)) or 0
        )
        row["source"] = "milvus"
        rows.append(row)

    return pd.DataFrame(rows)


def graph_keyword_search(query, limit=30):
    terms = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", query)
        if len(word) > 2
    ]

    if not terms:
        return pd.DataFrame()

    cypher = """
    MATCH (p:Property)
    WITH p, toLower(
        coalesce(toString(p.title), '') + ' ' +
        coalesce(toString(p.description), '') + ' ' +
        coalesce(toString(p.community), '')
    ) AS txt
    WHERE any(term IN $terms WHERE txt CONTAINS term)
    RETURN toString(p.property_id) AS property_id,
           p.title AS title,
           p.description AS description,
           p.price AS price,
           p.beds AS beds,
           p.bathrooms AS bathrooms,
           p.community AS community,
           p.property_type AS property_type,
           p.listing_type AS listing_type,
           p.status AS status,
           p.link AS link
    LIMIT $limit
    """

    rows = run_cypher(
        cypher,
        {"terms": terms, "limit": int(limit)},
    )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["source"] = "neo4j"
        frame["lexical_score"] = 1.0

    return frame


def graph_properties_by_ids(property_ids):
    ids = [
        str(value)
        for value in property_ids
        if str(value).strip()
    ]

    if not ids:
        return pd.DataFrame()

    cypher = """
    MATCH (p:Property)
    WHERE toString(p.property_id) IN $ids
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
    OPTIONAL MATCH (p)-[:IN_BUILDING]->(b:Building)
    OPTIONAL MATCH (p)-[:HAS_TYPE]->(pt:PropertyType)
    OPTIONAL MATCH (p)-[r:HAS_INTENT]->(ic:IntentCategory)
    RETURN toString(p.property_id) AS property_id,
           p.title AS title,
           p.description AS description,
           p.price AS price,
           p.beds AS beds,
           p.bathrooms AS bathrooms,
           coalesce(c.name, p.community) AS community,
           coalesce(pt.name, p.property_type) AS property_type,
           p.listing_type AS listing_type,
           p.status AS status,
           p.link AS link,
           b.name AS building,
           collect(DISTINCT {
               name: ic.name,
               evidence: r.evidence,
               source: r.source
           }) AS intents
    """

    return pd.DataFrame(
        run_cypher(cypher, {"ids": ids})
    )


def reciprocal_rank_fusion(frames, k=60):
    score_map = {}
    rows_by_id = {}

    for frame in frames:
        if frame is None or frame.empty:
            continue

        for rank, row in enumerate(
            frame.to_dict("records"), start=1
        ):
            pid = str(row.get("property_id", "")).strip()
            if not pid:
                continue

            score_map[pid] = (
                score_map.get(pid, 0.0) + 1.0 / (k + rank)
            )

            current = rows_by_id.setdefault(pid, {})

            for key, value in row.items():
                if value is None:
                    continue

                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass

                if key not in current or current[key] in (None, ""):
                    current[key] = value

    rows = []
    for pid, row in rows_by_id.items():
        row["property_id"] = pid
        row["fusion_score"] = score_map[pid]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        "fusion_score", ascending=False
    ).reset_index(drop=True)


def retrieve_hybrid(query, limit=30):
    dense = milvus_search(query, limit)
    lexical = graph_keyword_search(query, limit)

    fused = reciprocal_rank_fusion([dense, lexical])

    if fused.empty:
        return fused

    graph = graph_properties_by_ids(
        fused["property_id"].astype(str).tolist()
    )

    if graph.empty:
        return fused

    graph = graph.drop_duplicates("property_id")

    enriched = fused.merge(
        graph,
        on="property_id",
        how="left",
        suffixes=("", "_graph"),
    )

    # Preserve all dense/lexical hits, including those without graph nodes.
    for column in graph.columns:
        if column == "property_id":
            continue

        graph_column = f"{column}_graph"

        if graph_column in enriched.columns:
            if column in enriched.columns:
                enriched[column] = enriched[graph_column].combine_first(
                    enriched[column]
                )
            else:
                enriched[column] = enriched[graph_column]

            enriched.drop(columns=[graph_column], inplace=True)

    return enriched
'''),

    # 5. Hard gates and reranker
    markdown("""## 5. Hard gates and reranker
Enforce explicit constraints, then rerank candidates using retrieval relevance, intent tags, and soft preferences."""),
    code(r'''
def row_text(row):
    values = []

    for key in (
        "title", "description", "community",
        "property_type", "building",
    ):
        value = row.get(key)
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            values.append(str(value).lower())

    return " ".join(values)


def verify_hard_constraints(frame, plan):
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    hard = plan.get("hard_constraints") or {}

    transaction = plan.get("transaction_type")
    if transaction and "listing_type" in out:
        known = out["listing_type"].fillna("").astype(str).str.lower()
        mask = known.eq("") | known.str.contains(
            transaction, regex=False
        )
        out = out[mask]

    property_types = [
        value.lower()
        for value in clean_list(plan.get("property_types"))
    ]

    if property_types and "property_type" in out:
        values = out["property_type"].fillna("").astype(str).str.lower()
        out = out[
            values.eq("")
            | values.apply(
                lambda value: any(
                    wanted in value for wanted in property_types
                )
            )
        ]

    numeric_checks = [
        ("budget_min", "price", "min"),
        ("budget_max", "price", "max"),
        ("bedrooms_min", "beds", "min"),
        ("bedrooms_max", "beds", "max"),
        ("bathrooms_min", "bathrooms", "min"),
    ]

    for constraint, column, mode in numeric_checks:
        target = as_number(hard.get(constraint))

        if target is None or column not in out:
            continue

        values = pd.to_numeric(out[column], errors="coerce")
        condition = (
            values >= target if mode == "min"
            else values <= target
        )

        # Missing data remains eligible but should not be treated as verified.
        out = out[values.isna() | condition]

    hard_communities = [
        value.lower()
        for value in clean_list(hard.get("communities"))
    ]

    if hard_communities and "community" in out:
        values = out["community"].fillna("").astype(str).str.lower()
        out = out[
            values.eq("")
            | values.apply(
                lambda value: any(
                    name in value for name in hard_communities
                )
            )
        ]

    required = [
        value.lower()
        for value in clean_list(hard.get("required_features"))
    ]
    excluded = [
        value.lower()
        for value in clean_list(hard.get("excluded_features"))
    ]

    if required or excluded:
        def feature_ok(row):
            text = row_text(row)
            if not text:
                return True
            return (
                all(term in text for term in required)
                and not any(term in text for term in excluded)
            )

        out = out[out.apply(feature_ok, axis=1)]

    return out.reset_index(drop=True)


def intent_score(row, preferences):
    intents = row.get("intents")
    names = []

    if isinstance(intents, list):
        names = [
            str(item.get("name", "")).lower()
            for item in intents
            if isinstance(item, dict) and item.get("name")
        ]

    score = 0.0

    for category, weight in (preferences or {}).items():
        if category.lower() in names:
            try:
                score += max(0.0, min(1.0, float(weight)))
            except (TypeError, ValueError):
                pass

    return min(1.0, score)


def rerank_candidates(frame, plan, profile, top_k=10):
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()

    if "fusion_score" not in out:
        out["fusion_score"] = 0.0

    out["fusion_score"] = pd.to_numeric(
        out["fusion_score"], errors="coerce"
    ).fillna(0.0)

    preferences = (
        plan.get("intent_preferences")
        or profile.get("intent_preferences")
        or {}
    )

    out["intent_score"] = out.apply(
        lambda row: intent_score(row, preferences), axis=1
    )

    plan_soft = plan.get("soft_preferences") or {}
    profile_soft = profile.get("soft_preferences") or {}

    communities = [
        x.lower()
        for x in clean_list(
            plan_soft.get("preferred_communities")
        ) + clean_list(
            profile_soft.get("preferred_communities")
        )
    ]

    features = [
        x.lower()
        for x in clean_list(plan_soft.get("features"))
        + clean_list(profile_soft.get("features"))
    ]

    def preference_score(row):
        score = 0.0
        community = str(row.get("community", "")).lower()
        text = row_text(row)

        if communities and any(x in community for x in communities):
            score += 0.5

        if features:
            score += 0.5 * sum(
                feature in text for feature in features
            ) / len(features)

        return min(1.0, score)

    out["preference_score"] = out.apply(
        preference_score, axis=1
    )

    max_score = max(float(out["fusion_score"].max()), 1e-9)
    out["retrieval_score"] = out["fusion_score"] / max_score

    out["final_score"] = (
        0.65 * out["retrieval_score"]
        + 0.20 * out["intent_score"]
        + 0.15 * out["preference_score"]
    )

    return out.sort_values(
        "final_score", ascending=False
    ).head(top_k).reset_index(drop=True)


def community_summary(community):
    rows = run_cypher("""
        MATCH (p:Property)-[:LOCATED_IN]->(c:Community)
        WHERE toLower(c.name) CONTAINS toLower($community)
        RETURN c.name AS community,
               count(p) AS listing_count,
               avg(toFloat(p.price)) AS average_price,
               min(toFloat(p.price)) AS minimum_price,
               max(toFloat(p.price)) AS maximum_price,
               avg(toFloat(p.beds)) AS average_beds
    """, {"community": community})

    return pd.DataFrame(rows)


def portfolio_summary():
    rows = run_cypher("""
        MATCH (p:Property)
        RETURN count(p) AS listing_count,
               count(DISTINCT p.community) AS communities,
               avg(toFloat(p.price)) AS average_price,
               min(toFloat(p.price)) AS minimum_price,
               max(toFloat(p.price)) AS maximum_price
    """)

    return pd.DataFrame(rows)


def exact_property_lookup(property_id):
    rows = run_cypher("""
        MATCH (p:Property)
        WHERE toString(p.property_id) = $pid
        RETURN toString(p.property_id) AS property_id,
               p.title AS title,
               p.description AS description,
               p.price AS price,
               p.beds AS beds,
               p.bathrooms AS bathrooms,
               p.community AS community,
               p.property_type AS property_type,
               p.listing_type AS listing_type,
               p.status AS status,
               p.link AS link
        LIMIT 1
    """, {"pid": str(property_id)})

    return pd.DataFrame(rows)


def retrieve_candidates(plan, profile, top_k=10):
    query_type = plan.get("query_type", "property_search")

    if query_type == "exact_lookup":
        property_id = plan.get("exact_property_id")
        if property_id:
            frame = exact_property_lookup(property_id)
            if not frame.empty:
                frame["fusion_score"] = 1.0
                frame = verify_hard_constraints(frame, plan)
                return rerank_candidates(
                    frame, plan, profile, top_k
                )

    if query_type == "portfolio_analysis":
        return portfolio_summary()

    community = plan.get("community")
    normalized = plan.get("normalized_query", "").lower()

    if query_type == "community_analysis" and community:
        return community_summary(str(community))

    if community and any(
        term in normalized
        for term in ("community", "area", "neighborhood", "market")
    ):
        return community_summary(str(community))

    frame = retrieve_hybrid(
        plan.get("normalized_query", "")
    )
    frame = verify_hard_constraints(frame, plan)

    return rerank_candidates(
        frame, plan, profile, top_k
    )
'''),

    # 6. Grounded generation
    markdown("""## 6. Gated answer generation
Pass retrieved evidence to the answer model and require it to avoid unsupported property claims."""),
    code(r'''
ANSWER_SYSTEM_PROMPT = """
You are a careful real-estate research assistant.
Answer using ONLY the supplied evidence packet.
Never invent property details, availability, prices, amenities,
community statistics, or links.
If evidence is insufficient, say so.
Distinguish listing facts from calculated summaries.
Do not claim a property meets a requirement unless evidence supports it.
Keep the response clear for a non-technical customer.
"""


def json_safe_records(frame, limit=10):
    if frame is None or frame.empty:
        return []

    records = frame.head(limit).to_dict("records")
    safe = []

    for record in records:
        clean = {}

        for key, value in record.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = (
                    value
                    if not (
                        isinstance(value, float)
                        and not math.isfinite(value)
                    )
                    else None
                )
            elif isinstance(value, (list, dict)):
                clean[key] = value
            else:
                try:
                    clean[key] = value.item()
                except Exception:
                    clean[key] = str(value)

        safe.append(clean)

    return safe


def generate_grounded_answer(query, plan, profile, evidence):
    packet = {
        "user_query": query,
        "query_plan": plan,
        "customer_profile": profile,
        "retrieved_evidence": evidence,
    }

    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1300,
        temperature=0,
        system=ANSWER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    packet, ensure_ascii=False
                ),
            }
        ],
    )

    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()
'''),

    # 7. Pipeline orchestration
    markdown("""## 7. End-to-end orchestration and tracing
Connect query classification, profile updates, retrieval, hard gates, reranking, and grounded generation."""),
    code(r'''
def run_hybrid_pipeline(customer_id, query, history=None):
    customer_id = str(customer_id or "demo-customer").strip()
    query = str(query or "").strip()

    history = list(
        history if history is not None
        else HISTORY_STORE.get(customer_id, [])
    )

    if not query:
        return {
            "answer": "Please enter a question.",
            "results": pd.DataFrame(),
            "plan": {},
            "profile": PROFILE_STORE.get(
                customer_id, new_profile()
            ),
            "trace": {"error": "empty_query"},
            "history": history,
        }

    started = time.time()
    profile_before = PROFILE_STORE.get(
        customer_id, new_profile()
    )

    plan = classify_query(query, profile_before, history)

    profile_after = merge_profile(
        profile_before, plan.get("profile_updates", {})
    )
    PROFILE_STORE[customer_id] = profile_after

    retrieval_started = time.time()
    results = retrieve_candidates(
        plan, profile_after, top_k=10
    )
    retrieval_ms = round(
        (time.time() - retrieval_started) * 1000, 1
    )

    evidence = json_safe_records(results, limit=10)

    if not evidence:
        answer = (
            "I couldn't find sufficiently relevant evidence for "
            "that request. Try specifying a community, budget, "
            "property type, or property ID."
        )
    else:
        answer = generate_grounded_answer(
            query, plan, profile_after, evidence
        )

    history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])
    history = history[-12:]
    HISTORY_STORE[customer_id] = history

    trace = {
        "request_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "query_type": plan.get("query_type"),
        "retrieved_count": len(evidence),
        "retrieval_ms": retrieval_ms,
        "total_ms": round(
            (time.time() - started) * 1000, 1
        ),
        "profile_updated": profile_after != profile_before,
        "evidence_ids": [
            str(item["property_id"])
            for item in evidence
            if item.get("property_id")
        ],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return {
        "answer": answer,
        "results": results,
        "plan": plan,
        "profile": profile_after,
        "trace": trace,
        "history": history,
    }
'''),

    # 8. Smoke tests
    markdown("""## 8. Smoke tests
Verify profile patching and reciprocal-rank fusion without making LLM calls."""),
    code(r'''
test_profile = merge_profile(new_profile(), {
    "transaction_type": "sale",
    "hard_constraints": {
        "budget_max": 2500000,
        "bedrooms_min": 2,
    },
    "soft_preferences": {
        "preferred_communities": ["Dubai Marina"],
    },
})

assert test_profile["transaction_type"] == "sale"
assert test_profile["hard_constraints"]["budget_max"] == 2500000
assert "Dubai Marina" in test_profile[
    "soft_preferences"
]["preferred_communities"]

dense_test = pd.DataFrame([
    {"property_id": "A", "title": "Property A"},
    {"property_id": "B", "title": "Property B"},
])

lexical_test = pd.DataFrame([
    {"property_id": "B", "description": "Property B"},
    {"property_id": "C", "description": "Property C"},
])

fused_test = reciprocal_rank_fusion(
    [dense_test, lexical_test]
)

assert set(fused_test["property_id"]) == {"A", "B", "C"}

print("Smoke tests passed.")

try:
    count_result = run_cypher(
        "MATCH (p:Property) RETURN count(p) AS count"
    )
    print(
        "Neo4j Property nodes:",
        count_result[0]["count"] if count_result else "unknown",
    )
except Exception as exc:
    print("Neo4j check failed:", exc)
'''),

    # 9. Gradio interface
    markdown("""## 9. Gradio UI
Provide a simple interface for customer queries and inspectable results, query plan, profile, and trace."""),
    code(r'''
def ui_run(customer_id, query, history):
    try:
        output = run_hybrid_pipeline(
            customer_id, query, history
        )

        results = output["results"]

        desired_columns = [
            "property_id", "title", "price", "beds",
            "bathrooms", "community", "property_type",
            "listing_type", "final_score", "link",
        ]

        columns = [
            column for column in desired_columns
            if isinstance(results, pd.DataFrame)
            and column in results.columns
        ]

        table = (
            results[columns]
            if columns
            else pd.DataFrame()
        )

        return (
            output["answer"],
            table,
            json.dumps(
                output["plan"], indent=2,
                ensure_ascii=False, default=str
            ),
            json.dumps(
                output["profile"], indent=2,
                ensure_ascii=False, default=str
            ),
            json.dumps(
                output["trace"], indent=2,
                ensure_ascii=False, default=str
            ),
            output["history"],
        )

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return (
            f"Pipeline error: {message}",
            pd.DataFrame(),
            "{}",
            "{}",
            json.dumps({"error": message}, indent=2),
            history or [],
        )


with gr.Blocks(
    title="Hybrid Real-Estate Research Agent"
) as demo:
    gr.Markdown("# Hybrid Real-Estate Research Agent")
    gr.Markdown(
        "Query planning → profile update → hybrid retrieval "
        "→ hard gates → reranking → grounded answer"
    )

    with gr.Row():
        customer_id = gr.Textbox(
            label="Customer ID",
            value="demo-customer",
        )
        query = gr.Textbox(
            label="Question",
            placeholder=(
                "Find 2-bedroom apartments for sale in "
                "Dubai Marina under 2.5M"
            ),
            scale=3,
        )

    history_state = gr.State([])

    with gr.Row():
        run_button = gr.Button(
            "Search", variant="primary"
        )
        clear_button = gr.Button("Clear")

    answer = gr.Markdown()

    with gr.Tab("Ranked properties"):
        results_table = gr.Dataframe(
            label="Candidates",
            interactive=False,
            wrap=True,
        )

    with gr.Tab("Query plan"):
        plan_view = gr.Code(
            label="Query plan",
            language="json",
        )

    with gr.Tab("Customer profile"):
        profile_view = gr.Code(
            label="Updated profile",
            language="json",
        )

    with gr.Tab("Trace"):
        trace_view = gr.Code(
            label="Pipeline trace",
            language="json",
        )

    outputs = [
        answer, results_table, plan_view,
        profile_view, trace_view, history_state,
    ]

    run_button.click(
        fn=ui_run,
        inputs=[customer_id, query, history_state],
        outputs=outputs,
    )

    query.submit(
        fn=ui_run,
        inputs=[customer_id, query, history_state],
        outputs=outputs,
    )

    def clear_ui():
        return (
            "", pd.DataFrame(), "{}", "{}", "{}", []
        )

    clear_button.click(
        fn=clear_ui,
        inputs=[],
        outputs=outputs,
    )

print("Gradio UI ready.")
'''),

    # 10. Launch
    markdown("""## 10. Launch UI
Start the local Gradio interface."""),
    code(r'''
demo.launch(share=False, inbrowser=True)
'''),

    # 11. Cleanup
    markdown("""## 11. Cleanup
Close the Neo4j driver when finished, if needed."""),
    code(r'''
# neo4j_driver.close()
'''),
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Notebook 09."
    )
    parser.add_argument(
        "--output",
        default="notebooks/09_hybrid_rag_integration.ipynb",
        help="Notebook output path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Created notebook: {output.resolve()}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
