
from pathlib import Path
import json
import textwrap

OUTPUT = Path("09_hybrid_rag_end_to_end.ipynb")


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown("""
    # Real Estate Research Agent — End-to-End Notebook

    This notebook combines **Milvus semantic search, Neo4j structured
    retrieval, CSV listing enrichment, and Claude-generated answers**.

    ## How it works

    1. Claude interprets the question, selects a retrieval strategy,
       and normalizes filters using values discovered from the database.
    2. Milvus finds semantically relevant property IDs for descriptive queries.
    3. Neo4j retrieves authoritative property information and applies filters.
    4. CSV data supplies listing URLs, image URLs, and descriptions.
    5. Claude generates a grounded response with clickable listings.
    6. Gradio provides a simple interface for testing.

    **Important:** IDs connect the sources. Property facts come from the database;
    images and listing links come from the CSV when available.
    """),

    markdown("""
    ## 1. Install dependencies and connect services

    Configure your `.env` file before running the notebook. Existing environment
    variables are preserved. Do not put API keys directly in notebook cells.
    """),

    code(r'''
    %pip install -q anthropic neo4j pymilvus fastembed pandas python-dotenv gradio
    '''),

    code(r'''
    import os
    import re
    import json
    import ast
    import math
    import glob
    import pandas as pd
    import numpy as np

    from pathlib import Path
    from dotenv import load_dotenv
    from anthropic import Anthropic
    from neo4j import GraphDatabase
    from pymilvus import MilvusClient
    from fastembed import TextEmbedding
    from IPython.display import display

    load_dotenv()

    def env_required(name, *alternatives):
        for key in (name, *alternatives):
            value = os.getenv(key)
            if value:
                return value
        raise ValueError(
            f"Missing {name}. Add it to .env or your environment."
        )

    ANTHROPIC_API_KEY = env_required("ANTHROPIC_API_KEY")
    NEO4J_URI = env_required("NEO4J_URI")
    NEO4J_USERNAME = env_required("NEO4J_USERNAME", "NEO4J_USER")
    NEO4J_PASSWORD = env_required("NEO4J_PASSWORD")
    MILVUS_URI = env_required("MILVUS_URI")

    ANTHROPIC_MODEL = os.getenv(
        "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
    )
    MILVUS_COLLECTION = os.getenv(
        "MILVUS_COLLECTION", "property_listings"
    )
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    MILVUS_CANDIDATE_LIMIT = int(
        os.getenv("MILVUS_CANDIDATE_LIMIT", "30")
    )
    PROPERTY_CSV_PATH = os.getenv("PROPERTY_CSV_PATH", "")

    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    neo4j_driver.verify_connectivity()

    milvus_kwargs = {"uri": MILVUS_URI}
    if os.getenv("MILVUS_TOKEN"):
        milvus_kwargs["token"] = os.getenv("MILVUS_TOKEN")

    milvus_client = MilvusClient(**milvus_kwargs)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    print("Connections initialized.")
    print("Claude model:", ANTHROPIC_MODEL)
    print("Milvus collection:", MILVUS_COLLECTION)
    '''),

    markdown("""
    ## 2. Load and inspect the property CSV

    Set `PROPERTY_CSV_PATH` in `.env` if automatic discovery finds more than
    one candidate. The CSV enriches results; it is not used to invent facts.
    """),

    code(r'''
    def locate_property_csv():
        if PROPERTY_CSV_PATH:
            path = Path(PROPERTY_CSV_PATH).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"CSV not found: {path}")
            return path

        candidates = []
        for pattern in (
            "data/**/*.csv",
            "datasets/**/*.csv",
            "*.csv",
            "../data/**/*.csv",
        ):
            candidates.extend(Path(".").glob(pattern))

        candidates = list(dict.fromkeys(
            p.resolve() for p in candidates if p.is_file()
        ))

        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise FileNotFoundError(
                "No CSV found. Set PROPERTY_CSV_PATH in .env."
            )

        raise ValueError(
            "Multiple CSV files found. Set PROPERTY_CSV_PATH in .env:\n"
            + "\n".join(map(str, candidates[:20]))
        )

    csv_path = locate_property_csv()
    properties_df = pd.read_csv(csv_path, low_memory=False)
    properties_df.columns = [
        str(c).strip() for c in properties_df.columns
    ]

    print("CSV:", csv_path)
    print("Rows:", len(properties_df))
    print("Columns:", len(properties_df.columns))
    display(pd.DataFrame({
        "column": properties_df.columns,
        "non_null": [
            properties_df[c].notna().sum()
            for c in properties_df.columns
        ],
    }))
    '''),

    markdown("""
    ## 3. Normalize IDs and discover database vocabulary

    We discover the actual values used for listing type, property type,
    status, and community from Neo4j. The planner uses these values to avoid
    mismatches such as asking for `sale` when the database stores `purchase`.
    """),

    code(r'''
    def normalize_id(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        value = str(value).strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return None

        # Normalize numeric IDs stored as values such as 123.0.
        if re.fullmatch(r"\d+\.0+", value):
            value = value.split(".")[0]
        return value


    def find_column(df, candidates):
        lookup = {str(c).strip().lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        return None


    CSV_ID_COLUMN = find_column(
        properties_df,
        ["property_id", "id", "crm_id", "strapi_id", "listing_id"]
    )

    if not CSV_ID_COLUMN:
        raise ValueError(
            "Could not identify the CSV property ID column. "
            "Inspect the columns above and update CSV_ID_COLUMN."
        )

    properties_df["_join_id"] = properties_df[CSV_ID_COLUMN].map(
        normalize_id
    )
    properties_df = properties_df.dropna(subset=["_join_id"])
    properties_df = properties_df.drop_duplicates(
        subset=["_join_id"], keep="first"
    )
    properties_df = properties_df.set_index("_join_id", drop=False)

    def run_cypher(query, parameters=None):
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


    db_counts = run_cypher("""
        MATCH (p:Property)
        RETURN count(p) AS property_count
    """)
    print("Neo4j property count:", db_counts[0]["property_count"])

    vocabulary_rows = run_cypher("""
        MATCH (p:Property)
        OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
        RETURN
            collect(DISTINCT toString(p.listing_type)) AS listing_types,
            collect(DISTINCT toString(p.property_type)) AS property_types,
            collect(DISTINCT toString(p.status)) AS statuses,
            collect(DISTINCT toString(c.name)) AS communities
    """)

    vocabulary = vocabulary_rows[0] if vocabulary_rows else {}

    def clean_vocabulary(values):
        return sorted({
            str(v).strip()
            for v in (values or [])
            if v and str(v).strip().lower() not in
            {"nan", "none", "null"}
        })

    vocabulary = {
        key: clean_vocabulary(value)
        for key, value in vocabulary.items()
    }

    for key, values in vocabulary.items():
        print(f"{key}: {len(values)} distinct values")
        print(values[:20])

    print("CSV unique property IDs:", len(properties_df))
    '''),

    markdown("""
    ## 4. One LLM planning call

    Claude returns one structured plan. It chooses the retrieval strategy,
    extracts filters, and normalizes requested values against the actual
    database vocabulary.

    Strategy choices:
    - `semantic`: descriptive or similarity-based requests
    - `structured`: explicit filters or exact property lookup
    - `hybrid`: filters plus descriptive preferences
    - `analytics`: aggregate or community-level questions
    """),

    code(r'''
    PLANNER_PROMPT = """
    You are the query planner for a real-estate research system.

    Return ONLY valid JSON. Do not answer the user's property question.

    Database vocabulary:
    {vocabulary}

    Return this schema:
    {{
      "query_type": "property_search|exact_lookup|community_analysis|portfolio_analysis|general_real_estate",
      "retrieval_strategy": "semantic|structured|hybrid|analytics",
      "normalized_query": "clear normalized search request",
      "semantic_query": "description used for semantic similarity",
      "filters": {{
        "community": null,
        "property_type": null,
        "transaction_type": null,
        "status": null,
        "min_price": null,
        "max_price": null,
        "min_beds": null,
        "max_beds": null,
        "min_bathrooms": null,
        "max_bathrooms": null
      }},
      "exact_property_id": null,
      "community": null,
      "intent_preferences": [],
      "answer_mode": "listings|summary|analytics|explanation",
      "clarification_needed": false,
      "clarification_question": null
    }}

    Rules:
    1. Use only vocabulary values that actually exist in the supplied
       database vocabulary for community, property type, transaction type,
       and status. Match user wording to the closest actual value.
    2. For example, if the user says "sale" but the database vocabulary
       uses "purchase", use "purchase". Do not invent a new category.
    3. A property with a requested minimum or maximum must satisfy it.
    4. Use structured for exact IDs and mostly explicit filters.
    5. Use semantic for descriptive requests such as natural light,
       peaceful surroundings, or lifestyle fit without hard filters.
    6. Use hybrid when hard filters and descriptive preferences both matter.
    7. Use analytics for counts, averages, distributions, or community
       comparisons that need database aggregation.
    8. Put the original descriptive intent in semantic_query.
    9. Unknown numeric filters must be null, not guessed.
    10. Do not invent IDs, facts, communities, or listing details.
    11. If the question is too ambiguous to search meaningfully,
        set clarification_needed to true.
    12. Keep JSON values simple: numbers, strings, arrays, booleans, or null.
    """


    def parse_json_response(text):
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


    def plan_query(user_query):
        prompt = PLANNER_PROMPT.format(
            vocabulary=json.dumps(vocabulary, ensure_ascii=False)
        )
        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1200,
            system=prompt,
            messages=[{"role": "user", "content": user_query}],
        )
        raw = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        plan = parse_json_response(raw)

        required_keys = {
            "query_type", "retrieval_strategy", "filters",
            "normalized_query", "semantic_query"
        }
        missing = required_keys - set(plan)
        if missing:
            raise ValueError(f"Planner response missing keys: {missing}")

        if plan["retrieval_strategy"] not in {
            "semantic", "structured", "hybrid", "analytics"
        }:
            raise ValueError("Planner returned an unsupported strategy.")

        return plan


    example_plan = plan_query(
        "Show apartments in Dubai Marina under 2.5 million AED"
    )
    display(example_plan)
    '''),

    markdown("""
    ## 5. Retrieval helpers

    The embedding model converts descriptive queries into vectors.
    Milvus returns candidate property IDs. Neo4j retrieves facts by ID.
    We deliberately do not request `p.link` from Neo4j because listing URLs
    may exist only in the CSV.
    """),

    code(r'''
    def embed_query(query):
        vectors = list(embedding_model.query_embed(query))
        if not vectors:
            raise ValueError("Embedding model returned no vector.")
        return vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])


    def milvus_search_ids(query, limit=None):
        limit = limit or MILVUS_CANDIDATE_LIMIT
        vector = embed_query(query)

        results = milvus_client.search(
            collection_name=MILVUS_COLLECTION,
            data=[vector],
            limit=limit,
            output_fields=[
                "property_id", "title", "price", "beds",
                "bathrooms", "community", "property_type",
                "listing_type", "status"
            ],
        )

        hits = results[0] if results else []
        output = []
        for rank, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {}) or {}
            pid = normalize_id(
                entity.get("property_id", hit.get("id"))
            )
            if pid:
                output.append({
                    "property_id": pid,
                    "semantic_score": hit.get("distance"),
                    "semantic_rank": rank,
                })
        return output


    def neo4j_records_by_ids(ids):
        ids = list(dict.fromkeys(
            normalize_id(pid) for pid in ids if normalize_id(pid)
        ))
        if not ids:
            return []

        return run_cypher("""
            MATCH (p:Property)
            WHERE toString(p.property_id) IN $ids
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
            OPTIONAL MATCH (p)-[:IN_BUILDING]->(b:Building)
            OPTIONAL MATCH (p)-[:HAS_TYPE]->(pt:PropertyType)
            RETURN
                toString(p.property_id) AS property_id,
                p.title AS title,
                p.price AS price,
                p.beds AS beds,
                p.bathrooms AS bathrooms,
                coalesce(c.name, p.community) AS community,
                coalesce(pt.name, p.property_type) AS property_type,
                p.listing_type AS listing_type,
                p.status AS status,
                p.description AS description,
                p.latitude AS latitude,
                p.longitude AS longitude,
                b.name AS building
        """, {"ids": ids})


    def numeric_value(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            cleaned = str(value).replace(",", "").strip()
            number = float(cleaned)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError):
            return None


    def parse_image_urls(value):
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        if isinstance(value, (list, tuple)):
            items = value
        else:
            raw = str(value).strip()
            if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
                return []
            items = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    items = parser(raw)
                    break
                except (ValueError, SyntaxError, TypeError):
                    continue
            if items is None:
                items = [raw]

        if isinstance(items, dict):
            items = list(items.values())
        if not isinstance(items, (list, tuple)):
            items = [items]

        urls = []
        for item in items:
            if isinstance(item, dict):
                item = (
                    item.get("url") or item.get("src")
                    or item.get("original") or item.get("image")
                )
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                urls.append(item.strip())

        return list(dict.fromkeys(urls))


    IMAGE_COLUMN = find_column(
        properties_df,
        ["images", "image_urls", "image_url", "photos", "media"]
    )
    LINK_COLUMN = find_column(
        properties_df,
        ["link", "url", "listing_url", "property_url",
         "web_url", "permalink"]
    )
    TITLE_COLUMN = find_column(
        properties_df, ["title", "name", "property_title"]
    )
    DESCRIPTION_COLUMN = find_column(
        properties_df,
        ["long_description", "description", "summary"]
    )

    print("CSV image column:", IMAGE_COLUMN)
    print("CSV listing link column:", LINK_COLUMN)
    print("CSV title column:", TITLE_COLUMN)
    print("CSV description column:", DESCRIPTION_COLUMN)


    def enrich_with_csv(records):
        enriched = []
        for record in records:
            row = dict(record)
            pid = normalize_id(row.get("property_id"))

            if pid and pid in properties_df.index:
                source = properties_df.loc[pid]

                if isinstance(source, pd.DataFrame):
                    source = source.iloc[0]

                if IMAGE_COLUMN:
                    row["image_urls"] = parse_image_urls(
                        source.get(IMAGE_COLUMN)
                    )
                else:
                    row["image_urls"] = []

                if LINK_COLUMN:
                    url = source.get(LINK_COLUMN)
                    row["listing_url"] = (
                        str(url).strip()
                        if pd.notna(url) and str(url).startswith(("http://", "https://"))
                        else None
                    )
                else:
                    row["listing_url"] = None

                if TITLE_COLUMN and not row.get("title"):
                    row["title"] = source.get(TITLE_COLUMN)

                if DESCRIPTION_COLUMN and not row.get("description"):
                    row["description"] = source.get(DESCRIPTION_COLUMN)
            else:
                row["image_urls"] = []
                row["listing_url"] = None

            enriched.append(row)

        return enriched
    '''),

    markdown("""
    ## 6. Apply hard constraints and retrieve candidates

    Hard constraints are applied to authoritative property records. Unknown
    numeric values do not count as satisfying a requested numeric limit.

    Community matching is case-insensitive. The planner should normalize
    the requested community to the database vocabulary.
    """),

    code(r'''
    def as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


    def same_text(a, b):
        return str(a or "").strip().casefold() == str(b or "").strip().casefold()


    def apply_hard_filters(records, filters):
        result = []

        for row in records:
            community = row.get("community")
            ptype = row.get("property_type")
            listing_type = row.get("listing_type")
            status = row.get("status")

            requested_community = filters.get("community")
            if requested_community and not same_text(
                community, requested_community
            ):
                # Allow a partial community match only if the actual
                # database community contains the requested normalized name.
                if str(requested_community).casefold() not in str(
                    community or ""
                ).casefold():
                    continue

            if filters.get("property_type") and not same_text(
                ptype, filters["property_type"]
            ):
                continue

            if filters.get("transaction_type") and not same_text(
                listing_type, filters["transaction_type"]
            ):
                continue

            if filters.get("status") and not same_text(
                status, filters["status"]
            ):
                continue

            numeric_filters = [
                ("price", "min_price", "max_price"),
                ("beds", "min_beds", "max_beds"),
                ("bathrooms", "min_bathrooms", "max_bathrooms"),
            ]

            rejected = False
            for field, min_key, max_key in numeric_filters:
                minimum = numeric_value(filters.get(min_key))
                maximum = numeric_value(filters.get(max_key))

                if minimum is None and maximum is None:
                    continue

                actual = numeric_value(row.get(field))
                if actual is None:
                    rejected = True
                    break
                if minimum is not None and actual < minimum:
                    rejected = True
                    break
                if maximum is not None and actual > maximum:
                    rejected = True
                    break

            if not rejected:
                result.append(row)

        return result


    def structured_search(filters, limit=500):
        # Parameters are passed separately; user text is not inserted into Cypher.
        query = """
            MATCH (p:Property)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Community)
            OPTIONAL MATCH (p)-[:HAS_TYPE]->(pt:PropertyType)
            WITH p, c, pt,
                toLower(toString(coalesce(c.name, p.community, ''))) AS area,
                toLower(toString(coalesce(pt.name, p.property_type, ''))) AS ptype,
                toLower(toString(coalesce(p.listing_type, ''))) AS listing,
                toLower(toString(coalesce(p.status, ''))) AS state,
                toFloat(replace(toString(p.price), ',', '')) AS price,
                toFloat(toString(p.beds)) AS beds,
                toFloat(toString(p.bathrooms)) AS baths
            WHERE
                ($community IS NULL OR area CONTAINS toLower($community))
                AND ($property_type IS NULL OR ptype = toLower($property_type))
                AND ($transaction_type IS NULL OR listing = toLower($transaction_type))
                AND ($status IS NULL OR state = toLower($status))
                AND ($min_price IS NULL OR price >= $min_price)
                AND ($max_price IS NULL OR price <= $max_price)
                AND ($min_beds IS NULL OR beds >= $min_beds)
                AND ($max_beds IS NULL OR beds <= $max_beds)
                AND ($min_bathrooms IS NULL OR baths >= $min_bathrooms)
                AND ($max_bathrooms IS NULL OR baths <= $max_bathrooms)
            RETURN
                toString(p.property_id) AS property_id,
                p.title AS title,
                p.price AS price,
                p.beds AS beds,
                p.bathrooms AS bathrooms,
                coalesce(c.name, p.community) AS community,
                coalesce(pt.name, p.property_type) AS property_type,
                p.listing_type AS listing_type,
                p.status AS status,
                p.description AS description,
                p.latitude AS latitude,
                p.longitude AS longitude
            LIMIT $limit
        """

        params = {
            "community": filters.get("community"),
            "property_type": filters.get("property_type"),
            "transaction_type": filters.get("transaction_type"),
            "status": filters.get("status"),
            "min_price": numeric_value(filters.get("min_price")),
            "max_price": numeric_value(filters.get("max_price")),
            "min_beds": numeric_value(filters.get("min_beds")),
            "max_beds": numeric_value(filters.get("max_beds")),
            "min_bathrooms": numeric_value(filters.get("min_bathrooms")),
            "max_bathrooms": numeric_value(filters.get("max_bathrooms")),
            "limit": int(limit),
        }
        return run_cypher(query, params)


    def exact_property_lookup(property_id):
        pid = normalize_id(property_id)
        if not pid:
            return []
        return neo4j_records_by_ids([pid])


    def community_analytics(community):
        return run_cypher("""
            MATCH (p:Property)-[:LOCATED_IN]->(c:Community)
            WHERE toLower(toString(c.name)) CONTAINS toLower($community)
            WITH c, p,
                toFloat(replace(toString(p.price), ',', '')) AS price
            RETURN
                c.name AS community,
                count(p) AS listing_count,
                avg(price) AS average_price,
                min(price) AS minimum_price,
                max(price) AS maximum_price
        """, {"community": community})


    def retrieve_candidates(plan, top_k=TOP_K):
        strategy = plan["retrieval_strategy"]
        filters = plan.get("filters") or {}
        diagnostics = {"strategy": strategy}

        if plan.get("clarification_needed"):
            return [], diagnostics

        if strategy == "analytics":
            community = plan.get("community") or filters.get("community")
            if community:
                rows = community_analytics(community)
                diagnostics["analytics_rows"] = len(rows)
                return rows, diagnostics

            rows = run_cypher("""
                MATCH (p:Property)
                RETURN
                    count(p) AS listing_count,
                    avg(toFloat(replace(toString(p.price), ',', ''))) AS average_price,
                    min(toFloat(replace(toString(p.price), ',', ''))) AS minimum_price,
                    max(toFloat(replace(toString(p.price), ',', ''))) AS maximum_price
            """)
            diagnostics["analytics_rows"] = len(rows)
            return rows, diagnostics

        if strategy == "semantic":
            hits = milvus_search_ids(
                plan.get("semantic_query") or plan["normalized_query"]
            )
            ids = [hit["property_id"] for hit in hits]
            records = neo4j_records_by_ids(ids)
            diagnostics["milvus_hits"] = len(hits)
            diagnostics["neo4j_records"] = len(records)
            candidates = apply_hard_filters(records, filters)

        elif strategy == "structured":
            if plan.get("exact_property_id"):
                records = exact_property_lookup(plan["exact_property_id"])
            else:
                records = structured_search(filters, limit=1000)
            diagnostics["structured_records"] = len(records)
            candidates = apply_hard_filters(records, filters)

        elif strategy == "hybrid":
            # First establish eligible properties using explicit constraints.
            eligible = apply_hard_filters(
                structured_search(filters, limit=2000), filters
            )
            eligible_by_id = {
                normalize_id(r.get("property_id")): r
                for r in eligible if normalize_id(r.get("property_id"))
            }

            hits = milvus_search_ids(
                plan.get("semantic_query") or plan["normalized_query"]
            )
            hit_by_id = {
                h["property_id"]: h for h in hits
            }

            # Preserve semantic ranking, but never return an ineligible property.
            candidates = [
                eligible_by_id[pid]
                for pid in hit_by_id if pid in eligible_by_id
            ]

            # If semantic retrieval found no eligible match, return the
            # structured eligible set rather than incorrectly returning nothing.
            if not candidates:
                candidates = eligible

            diagnostics["eligible_records"] = len(eligible)
            diagnostics["milvus_hits"] = len(hits)
            diagnostics["semantic_eligible_matches"] = sum(
                pid in eligible_by_id for pid in hit_by_id
            )

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        diagnostics["before_enrichment"] = len(candidates)
        candidates = enrich_with_csv(candidates)

        # Put semantic matches first when semantic scores are available.
        if strategy in {"semantic", "hybrid"}:
            hit_order = {
                h["property_id"]: i
                for i, h in enumerate(
                    milvus_search_ids(
                        plan.get("semantic_query") or plan["normalized_query"]
                    )
                )
            }
            candidates.sort(
                key=lambda row: hit_order.get(
                    normalize_id(row.get("property_id")), 10**9
                )
            )

        diagnostics["after_enrichment"] = len(candidates)
        return candidates[:top_k], diagnostics
    '''),

    markdown("""
    ## 7. Grounded answer generation

    Claude receives only retrieved records. It must not create property IDs,
    prices, locations, images, or listing links. The interface separately
    displays the actual image URLs and listing links from the enriched records.
    """),

    code(r'''
    ANSWER_PROMPT = """
    You are a careful real-estate research assistant.

    Answer using ONLY the supplied evidence.
    Do not invent or infer missing property facts.
    If no matching listings are available, say so clearly.
    Distinguish facts from recommendations.
    Use simple, readable language.
    For each property, mention its ID, price, bedrooms, bathrooms, property
    type, and community only when those fields are available.
    Do not fabricate URLs or image links.
    """


    def generate_answer(question, plan, records):
        if plan.get("clarification_needed"):
            return plan.get("clarification_question") or (
                "Could you clarify what kind of property you are looking for?"
            )

        if not records:
            return (
                "I couldn't find matching records in the connected data. "
                "Try adjusting the filters or broadening the search."
            )

        evidence = []
        for row in records:
            evidence.append({
                "property_id": row.get("property_id"),
                "title": row.get("title"),
                "price": row.get("price"),
                "beds": row.get("beds"),
                "bathrooms": row.get("bathrooms"),
                "community": row.get("community"),
                "property_type": row.get("property_type"),
                "listing_type": row.get("listing_type"),
                "status": row.get("status"),
                "description": row.get("description"),
                "building": row.get("building"),
            })

        response = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1800,
            system=ANSWER_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Normalized plan:\n{json.dumps(plan, default=str)}\n\n"
                    f"Retrieved evidence:\n{json.dumps(evidence, default=str)}"
                ),
            }],
        )

        return "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )


    def search_properties(question, top_k=TOP_K):
        plan = plan_query(question)
        records, diagnostics = retrieve_candidates(plan, top_k=top_k)
        answer = generate_answer(question, plan, records)

        return {
            "question": question,
            "plan": plan,
            "answer": answer,
            "records": records,
            "diagnostics": diagnostics,
        }
    '''),

    markdown("""
    ## 8. Run example queries

    Try a structured search, a descriptive search, a hybrid search,
    and an analytics question. Review the plan and diagnostics to see which
    path was used.
    """),

    code(r'''
    test_queries = [
        "Show apartments in Dubai Marina under 2.5 million AED",
        "Find a peaceful home with lots of natural light",
        "Find family-friendly apartments in Dubai Marina under 3 million AED",
        "How many properties are listed in Dubai Marina, and what is the average price?",
    ]

    for question in test_queries:
        print("\\n" + "=" * 90)
        print("QUESTION:", question)
        try:
            result = search_properties(question)
            print("\\nPLAN:")
            print(json.dumps(result["plan"], indent=2, default=str))
            print("\\nANSWER:")
            print(result["answer"])
            print("\\nDIAGNOSTICS:")
            print(json.dumps(result["diagnostics"], indent=2, default=str))
            print("Returned listings:", len(result["records"]))
        except Exception as exc:
            print("Query failed:", repr(exc))
    '''),

    markdown("""
    ## 9. Display property cards, images, and listing links

    The following table and gallery use the enriched records. A gallery image
    appears only when a real image URL exists in the CSV. A listing link appears
    only when a valid URL was found.
    """),

    code(r'''
    def results_dataframe(records):
        rows = []
        for r in records:
            rows.append({
                "property_id": r.get("property_id"),
                "title": r.get("title"),
                "price": r.get("price"),
                "beds": r.get("beds"),
                "bathrooms": r.get("bathrooms"),
                "community": r.get("community"),
                "property_type": r.get("property_type"),
                "listing_type": r.get("listing_type"),
                "listing_url": r.get("listing_url"),
                "image_count": len(r.get("image_urls") or []),
            })
        return pd.DataFrame(rows)


    def gallery_items(records):
        items = []
        for r in records:
            urls = r.get("image_urls") or []
            caption = (
                f"ID {r.get('property_id')} | "
                f"{r.get('title') or 'Property'} | "
                f"Price: {r.get('price') or 'Not available'}"
            )
            for url in urls[:3]:
                items.append((url, caption))
        return items


    # Run one query and inspect all returned details.
    demo = search_properties(
        "Show apartments in Dubai Marina under 2.5 million AED"
    )
    print(demo["answer"])
    display(results_dataframe(demo["records"]))
    print("Gallery images:", len(gallery_items(demo["records"])))
    '''),

    markdown("""
    ## 10. Gradio interface

    The UI shows the grounded answer, a table of matching listings, and a
    gallery for real image URLs. Missing images or links remain missing;
    the notebook does not fabricate them.
    """),

    code(r'''
    import gradio as gr


    def gradio_search(question):
        if not question or not question.strip():
            return "Enter a property question.", pd.DataFrame(), []

        try:
            result = search_properties(question.strip())
            table = results_dataframe(result["records"])
            gallery = gallery_items(result["records"])

            answer = result["answer"]
            answer += "\n\n---\n**Retrieval diagnostics**\n```json\n"
            answer += json.dumps(
                result["diagnostics"], indent=2, default=str
            )
            answer += "\n```"

            if not result["records"]:
                answer += (
                    "\n\nNo property cards were returned. "
                    "Check the planner and diagnostics above."
                )

            return answer, table, gallery

        except Exception as exc:
            return (
                f"Search failed: {type(exc).__name__}: {exc}",
                pd.DataFrame(),
                [],
            )


    demo_app = gr.Interface(
        fn=gradio_search,
        inputs=gr.Textbox(
            label="Ask about properties",
            placeholder=(
                "e.g. Find family-friendly apartments in Dubai Marina "
                "under 3 million AED"
            ),
            lines=2,
        ),
        outputs=[
            gr.Markdown(label="Answer and diagnostics"),
            gr.Dataframe(label="Property results", wrap=True),
            gr.Gallery(
                label="Property images",
                columns=3,
                height="auto",
                object_fit="contain",
            ),
        ],
        title="Real Estate Research Assistant",
        description=(
            "Search property records using semantic retrieval, structured "
            "filters, or a combination of both."
        ),
        examples=[
            ["Show apartments in Dubai Marina under 2.5 million AED"],
            ["Find a peaceful home with lots of natural light"],
            ["Find family-friendly apartments in Dubai Marina"],
            ["How many properties are listed in Dubai Marina?"],
        ],
    )

    # Set share=True only if you intentionally want a public temporary URL.
    demo_app.launch(share=False)
    '''),

    markdown("""
    ## 11. Troubleshooting checklist

    - **Zero results:** inspect the plan, database vocabulary, and diagnostics.
      Verify that `sale` was normalized to the actual stored listing type.
    - **Milvus errors:** check `MILVUS_URI`, token, collection name, vector
      dimension, and the embedding model used to build the collection.
    - **Neo4j errors:** check credentials, database name, property IDs,
      labels, and relationship names.
    - **Missing images or listing links:** verify the CSV columns and ensure
      CSV IDs match Neo4j/Milvus property IDs.
    - **Missing image cards:** check that `images` contains valid HTTP(S)
      URLs and that the URLs are publicly reachable by the UI.
    - **Planner mismatch:** inspect the discovered vocabulary and the plan.
      Avoid adding arbitrary aliases before confirming the stored values.

    When finished, close the Neo4j connection with:
    `neo4j_driver.close()`
    """),
]


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
            "version": "3.10+",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Notebook created: {OUTPUT.resolve()}")
print(f"Total cells: {len(cells)}")
