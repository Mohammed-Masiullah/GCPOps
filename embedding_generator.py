import json
import time
from typing import Dict, List, Iterable, Tuple
from google.cloud import bigquery, storage
from google.cloud.bigquery import Table, SchemaField
from google.api_core.exceptions import NotFound, ResourceExhausted

# --- ASSUMED EXTERNAL INITIALIZATION ---
# bq_client = bigquery.Client(project=PROJECT_ID)
# storage_client = storage.Client(project=PROJECT_ID)
# vertex_client = ... (initialized Vertex AI client for models)

# --- GLOBAL CONFIGURATION (From your input) ---
PROJECT_ID = "indigo-night-477403-b2"
REGION = "us-central1"
BUCKET_NAME = "movie-data-embedding"
GCS_PREFIX = ""
DATASET_ID = "movie_embeddings_dataset"
TABLE_ID = "movies_vectors"
MODEL_NAME = "gemini-embedding-001"
CHUNK_MAX_CHARS = 2500
CHUNK_OVERLAP = 200
# NOTE: Reducing this is critical for avoiding 429 errors.
BATCH_SIZE_EMBED = 5 # Adjusted down from 20 for stability
BATCH_SIZE_INSERT = 500
MAX_RECORDS_TO_PROCESS = None
USE_LOAD_JOB_FOR_BIGQUERY = False
BIGQUERY_LOCATION = "US"


# --- Helper Functions (No changes needed, included for completeness) ---

def list_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """Helper function to list JSON files in GCS."""
    # Placeholder for the actual listing logic, assuming it returns blob names
    return [blob.name for blob in storage_client.list_blobs(bucket_name, prefix=prefix) if blob.name.endswith('.json')]

def stream_json_records_from_gcs(bucket_name: str, blob_name: str) -> Iterable[Dict]:
    """Streams and parses records from a GCS JSON file (NDJSON or single array)."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    stream = blob.open("r")
    first = stream.readline()
    if not first:
        return
    first_stripped = first.strip()

    # Handle NDJSON
    if first_stripped.startswith("{") and not first_stripped.startswith("["):
        try:
            yield json.loads(first)
        except Exception:
            pass
        for line in stream:
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    # Handle JSON array (loads entire file)
    elif first_stripped.startswith("["):
        rest = stream.read()
        all_text = first + rest
        try:
            arr = json.loads(all_text)
            for item in arr:
                yield item
        except Exception as e:
            # Fallback for large array streaming is complex and often custom; simplified to print warning here
            print(f"Warning: failed to parse JSON array for {blob_name}: {e}. Fallback logic skipped.")
    else:
        stream.seek(0)
        text = stream.read()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    yield item
            elif isinstance(parsed, dict):
                yield parsed
        except Exception as e:
            print(f"Could not parse {blob_name}: {e}")

def flatten_movie_record(record: Dict) -> Tuple[str, str, Dict]:
    """Extracts text blob and metadata from movie record."""
    # (Existing detailed logic to flatten and format movie data)
    # This remains unchanged, returning (movie_id, text_blob, metadata)
    movie_id = record.get("id") or record.get("tconst") or None
    title = None
    if isinstance(record.get("titleText"), dict):
        title = record["titleText"].get("text")
    elif isinstance(record.get("title"), str):
        title = record.get("title")

    year = None
    if isinstance(record.get("releaseYear"), dict):
        year = record.get("releaseYear", {}).get("year")
    elif "releaseYear" in record:
        year = record.get("releaseYear")

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if year:
        parts.append(f"Year: {year}")

    genres_list = []
    if isinstance(record.get("genres"), dict):
        arr = record["genres"].get("genres") or []
        for g in arr:
            if isinstance(g, dict) and g.get("text"):
                genres_list.append(g.get("text"))
            elif isinstance(g, str):
                genres_list.append(g)
    if genres_list:
        parts.append("Genres: " + ", ".join(genres_list))

    rs = record.get("ratingsSummary") or {}
    rating = rs.get("aggregateRating")
    votes = rs.get("voteCount")
    if rating:
        parts.append(f"IMDb rating: {rating} (votes: {votes})")

    if record.get("metacritic"):
        parts.append(f"Metacritic: {record.get('metacritic')}")

    plot = None
    if isinstance(record.get("plot"), dict):
        plot = record["plot"].get("text") or str(record["plot"])
    elif isinstance(record.get("plot"), str):
        plot = record["plot"]
    if plot:
        parts.append("Plot: " + plot)

    cert = record.get("certificate")
    if cert:
        parts.append("Certificate: " + str(cert))
    spoken = record.get("spokenLanguages")
    if spoken:
        if isinstance(spoken, list):
            langs = []
            for lang in spoken:
                if isinstance(lang, dict) and lang.get("text"):
                    langs.append(lang["text"])
                elif isinstance(lang, str):
                    langs.append(lang)
            if langs:
                parts.append("Languages: " + ", ".join(langs))

    countries = record.get("countriesOfOrigin")
    if countries:
        if isinstance(countries, list):
            c = []
            for cc in countries:
                if isinstance(cc, dict) and cc.get("text"):
                    c.append(cc["text"])
                elif isinstance(cc, str):
                    c.append(cc)
            if c:
                parts.append("Countries: " + ", ".join(c))

    pc = record.get("principalCredits") or []
    if pc and isinstance(pc, list):
        directors = []
        cast = []
        writers = []
        for entry in pc:
            cat = entry.get("category", {}).get("text") if isinstance(entry.get("category"), dict) else None
            credits = entry.get("credits") or []
            names = []
            for c in credits:
                name = None
                if isinstance(c.get("name"), dict):
                    name = c["name"].get("nameText", {}).get("text")
                elif isinstance(c.get("name"), str):
                    name = c.get("name")
                if name:
                    names.append(name)
            if cat:
                if cat.lower() == "director":
                    directors.extend(names)
                elif cat.lower() == "writer":
                    writers.extend(names)
                elif cat.lower() in ("actor", "cast", "principal cast"):
                    cast.extend(names)
        if directors:
            parts.append("Directors: " + ", ".join(directors))
        if writers:
            parts.append("Writers: " + ", ".join(writers))
        if cast:
            parts.append("Main cast: " + ", ".join(cast))

    keywords = record.get("keywords")
    if keywords:
        if isinstance(keywords, list):
            parts.append("Keywords: " + ", ".join([kw.get("text") if isinstance(kw, dict) else str(kw) for kw in keywords[:20]]))

    companies = record.get("companyCredits")
    if companies:
        parts.append("Production companies: " + str(companies))

    if record.get("meterRanking"):
        parts.append(f"Meter ranking: {record.get('meterRanking')}")
    if record.get("isAdult") is not None:
        parts.append(f"isAdult: {record.get('isAdult')}")

    text_blob = "\n\n".join(parts)
    metadata = {"title": title, "year": year, "genres": genres_list, "movie_id": movie_id}
    return movie_id, text_blob, metadata

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into chunks with overlap."""
    if not text:
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# --- FIX 2: Updated embed_texts with Exponential Backoff ---

# def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
#     """
#     Generates embeddings for texts with exponential backoff retry on 429 errors.
#     Returns list of embedding vectors.
#     """
#     MAX_RETRIES = 5
#     BASE_DELAY = 2.0  # seconds

#     for attempt in range(MAX_RETRIES):
#         try:
#             # Attempt batch call
#             response = vertex_client.models.embed_content(model=model_name, contents=texts)

#             # Success - return list of embedding values
#             embeddings_out = [emb.values for emb in response.embeddings]
#             return embeddings_out

#         except ResourceExhausted as e:
#             if attempt < MAX_RETRIES - 1:
#                 # Calculate exponential backoff time
#                 wait_time = BASE_DELAY * (2 ** attempt)
#                 print(f"Quota exhausted (429). Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
#                 time.sleep(wait_time)
#             else:
#                 # Final failure after all retries
#                 print(f"FATAL: Quota exhausted after {MAX_RETRIES} attempts. Cannot embed texts. Error: {e}")
#                 # Return empty vectors for all texts in this failed batch
#                 return [[] for _ in texts]

#         except Exception as e:
#             # Handle other non-retryable errors (e.g., if content is too large)
#             print(f"Embedding error (non-quota issue): {e}. Inserting empty vectors for this batch.")
#             return [[] for _ in texts]

#     return [[] for _ in texts]

import time
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Generates embeddings for texts with exponential backoff on quota (429) or rate limit errors.
    Returns list of embedding vectors or [] for failed embeddings.
    """
    MAX_RETRIES = 6
    BASE_DELAY = 2.0  # seconds

    for attempt in range(MAX_RETRIES):
        try:
            response = vertex_client.models.embed_content(
                model=model_name,
                contents=texts
            )
            return [emb.values for emb in response.embeddings]

        except (ResourceExhausted, GoogleAPICallError) as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = BASE_DELAY * (2 ** attempt)
                print(f"Quota exhausted (429). Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue  # retry
            else:
                print(f"Embedding error (non-retryable): {e}")
                break

        except Exception as e:
            print(f"Unexpected embedding error: {e}")
            break

    print(f"Failed after {MAX_RETRIES} attempts. Returning empty vectors.")
    return [[] for _ in texts]


# --- Updated ensure_bq_dataset_and_table (Robust BQ check/creation) ---

# def ensure_bq_dataset_and_table(project: str, dataset_id: str, table_id: str, embedding_dim: int = None):
#     """
#     Ensures the BigQuery dataset and table (with vector embedding schema) exist.
#     """
#     # Use the client's method for correct DatasetReference instantiation
#     dataset_ref = bq_client.dataset(dataset_id)

#     # 1. Dataset Check/Creation
#     try:
#         bq_client.get_dataset(dataset_ref)
#         print(f"Dataset {dataset_id} exists.")
#     except NotFound:
#         print(f"Creating dataset {dataset_id}")
#         dataset = bigquery.Dataset(dataset_ref)
#         dataset.location = BIGQUERY_LOCATION
#         bq_client.create_dataset(dataset)
#         print(f"Dataset {dataset_id} created.")

#     # 2. Table Check/Creation
#     table_ref = dataset_ref.table(table_id)

#     try:
#         bq_client.get_table(table_ref)
#         print(f"Table {table_id} already exists.")
#     except NotFound:
#         print(f"Creating table {table_id}")
#         schema = [
#             SchemaField("movie_id", "STRING"),
#             SchemaField("title", "STRING"),
#             SchemaField("chunk_id", "STRING"),
#             SchemaField("chunk_text", "STRING"),
#             # ARRAY<FLOAT64> schema definition
#             SchemaField("embedding", "FLOAT64", mode="REPEATED"),
#             SchemaField("metadata_json", "STRING"),
#             SchemaField("source_file", "STRING")
#         ]

#         table = Table(table_ref, schema=schema)
#         bq_client.create_table(table)
#         print("Created table.")

def ensure_bq_dataset_and_table(project: str, dataset_id: str, table_id: str, embedding_dim: int = None):
    """
    Ensures the BigQuery dataset and table (with vector embedding schema) exist.
    Waits for confirmation after creation to avoid 404 insert errors.
    """
    dataset_ref = bq_client.dataset(dataset_id)

    # --- 1. Dataset Check/Creation ---
    try:
      bq_client.get_dataset(dataset_ref)
      print(f"Dataset {dataset_id} exists.")
    except NotFound:
      print(f"Creating dataset {dataset_id}")
      dataset = bigquery.Dataset(dataset_ref)
      dataset.location = BIGQUERY_LOCATION
      bq_client.create_dataset(dataset)
      print(f"Dataset {dataset_id} created.")

      # ✅ Wait for dataset to propagate
      for i in range(10):
        try:
          bq_client.get_dataset(dataset_ref)
          print(f"Dataset {dataset_id} is now available.")
          break
        except NotFound:
          wait_time = 2 ** i
          print(f"Waiting {wait_time}s for dataset {dataset_id} to propagate...")
          time.sleep(wait_time)
      else:
        raise RuntimeError(f"Dataset {dataset_id} did not become available after retries.")

    # --- 2. Table Check/Creation ---
    table_ref = dataset_ref.table(table_id)
    try:
        bq_client.get_table(table_ref)
        print(f"Table {table_id} already exists.")
    except NotFound:
        print(f"Creating table {table_id}")
        schema = [
            SchemaField("movie_id", "STRING"),
            SchemaField("title", "STRING"),
            SchemaField("chunk_id", "STRING"),
            SchemaField("chunk_text", "STRING"),
            SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            SchemaField("metadata_json", "STRING"),
            SchemaField("source_file", "STRING"),
        ]
        table = Table(table_ref, schema=schema)
        bq_client.create_table(table)
        print(f"Table {table_id} created.")

        # ✅ Wait and confirm table exists
        for i in range(10):
            try:
                bq_client.get_table(table_ref)
                print(f"Table {table_id} is now available.")
                break
            except NotFound:
                wait_time = 2 ** i
                print(f"Waiting {wait_time}s for table {table_id} to propagate...")
                time.sleep(wait_time)
        else:
            raise RuntimeError(f"Table {table_id} did not become available after retries.")


def insert_rows_to_bq(project: str, dataset_id: str, table_id: str, rows: List[Dict]):
    """Inserts rows using streaming API."""
    table_ref = f"{project}.{dataset_id}.{table_id}"
    # errors = bq_client.insert_rows_json(table_ref, rows)
    job = bq_client.load_table_from_json(rows, table_ref)
    return job.result()

    if job:
        print("Insert errors: ", job.result())
    # return errors

def process_all_files():
    files = list_gcs_files(BUCKET_NAME, GCS_PREFIX)
    print(f"Found {len(files)} json files in gs://{BUCKET_NAME}/{GCS_PREFIX}")
    processed = 0
    first_embedding_dim = None
    bq_resources_checked = False  # New flag to ensure BQ check happens only once

    for blob_name in files:
        print("Processing:", blob_name)

        for record in stream_json_records_from_gcs(BUCKET_NAME, blob_name):
            if MAX_RECORDS_TO_PROCESS and processed >= MAX_RECORDS_TO_PROCESS:
                print("Reached processing limit.")
                return
            processed += 1

            movie_id, text_blob, metadata = flatten_movie_record(record)
            if not text_blob:
                continue

            chunks = chunk_text(text_blob, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP)
            chunk_objs = []
            for idx, c in enumerate(chunks):
                chunk_id = f"{movie_id}_chunk_{idx}"
                chunk_objs.append({"movie_id": movie_id, "title": metadata.get("title"), "chunk_id": chunk_id, "chunk_text": c, "metadata_json": json.dumps(metadata), "source_file": blob_name})

            # Embed in groups
            for i in range(0, len(chunk_objs), BATCH_SIZE_EMBED):
                batch = chunk_objs[i:i+BATCH_SIZE_EMBED]
                texts = [x["chunk_text"] for x in batch]

                # Use the new embed_texts with retry logic (Fix 2)
                embeddings = embed_texts(texts, model_name=MODEL_NAME)

                rows = []
                for j, emb in enumerate(embeddings):
                    if not emb:
                        print(f"Skipping chunk {batch[j]['chunk_id']} due to empty embedding.")
                        continue

                    row = {
                        "movie_id": batch[j]["movie_id"],
                        "title": batch[j]["title"] or "",
                        "chunk_id": batch[j]["chunk_id"],
                        "chunk_text": batch[j]["chunk_text"],
                        "embedding": emb,
                        "metadata_json": batch[j]["metadata_json"],
                        "source_file": batch[j]["source_file"]
                    }
                    rows.append(row)

                    # Check/Set first_embedding_dim ONCE
                    if first_embedding_dim is None and emb:
                        first_embedding_dim = len(emb)

                # Check/Create BQ resources only ONCE after getting the first dimension (Fix 1)
                if not bq_resources_checked and first_embedding_dim is not None:
                    ensure_bq_dataset_and_table(PROJECT_ID, DATASET_ID, TABLE_ID, embedding_dim=first_embedding_dim)
                    bq_resources_checked = True

                # Insert in batches to BigQuery (only if table check was successful and we have rows)
                if bq_resources_checked and rows:
                    for k in range(0, len(rows), BATCH_SIZE_INSERT):
                        chunk_rows = rows[k:k+BATCH_SIZE_INSERT]
                        insert_rows_to_bq(PROJECT_ID, DATASET_ID, TABLE_ID, chunk_rows)

        print(f"Finished file: {blob_name}")

    print("Done. Processed total records:", processed)

# --- Execution starts here (assuming you run this line) ---
# process_all_files()