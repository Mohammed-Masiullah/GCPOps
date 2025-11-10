import time
from typing import List, Dict, Tuple, Optional
from google.cloud import bigquery
from google import genai
from google.genai.types import Part, GenerateContentConfig
from sentence_transformers import CrossEncoder
import json


PROJECT_ID = "indigo-night-477403-b2"
REGION = "us-central1"
DATASET_ID = "movie_embeddings_dataset"
TABLE_ID = "movies_vectors"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
INITIAL_TOP_K = 50  # Cast a wide net initially
RERANK_TOP_K = 8    # Final number of chunks to pass to LLM
MAX_CONTEXT_TOKENS = 30000  # Conservative limit for better performance
RESERVED_OUTPUT_TOKENS = 4000  # Reserve space for response

bq_client = bigquery.Client(project=PROJECT_ID)
genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=REGION
)

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove near-duplicate chunks and aggregate information from same movie.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Deduplicated list of chunks
    """
    seen_movies = {}
    deduplicated = []
    
    for chunk in chunks:
        movie_id = chunk['movie_id']
        
        # Keep best chunk per movie (already sorted by relevance)
        if movie_id not in seen_movies:
            seen_movies[movie_id] = True
            deduplicated.append(chunk)
    
    if len(deduplicated) < len(chunks):
        print(f"  Deduplicated: {len(chunks)} → {len(deduplicated)} chunks")
    
    return deduplicated


# --- STEP 4: TOKEN MANAGEMENT ---

def count_tokens(text: str, model: str = LLM_MODEL) -> int:
    """
    Count tokens in text using Gemini's count_tokens API.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for tokenization
        
    Returns:
        Number of tokens
    """
    try:
        response = genai_client.models.count_tokens(
            model=model,
            contents=[text]
        )
        return response.total_tokens
    except Exception as e:
        # Fallback: rough estimate (1 token ≈ 4 characters for English)
        print(f"Warning: Token counting failed, using estimate ({e})")
        return len(text) // 4


def select_chunks_within_token_budget(
    chunks: List[Dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    model: str = LLM_MODEL
) -> List[Dict]:
    """
    Select chunks that fit within token budget using greedy approach.
    
    Iteratively adds highest-scored chunks until budget is exhausted.
    
    Args:
        chunks: List of chunks sorted by relevance
        max_tokens: Maximum tokens allowed
        model: Model name for tokenization
        
    Returns:
        List of selected chunks that fit within budget
    """
    selected = []
    total_tokens = 0
    
    print(f"📊 Selecting chunks within {max_tokens:,} token budget...")
    
    for chunk in chunks:
        chunk_tokens = count_tokens(chunk['chunk_text'], model)
        
        if total_tokens + chunk_tokens <= max_tokens:
            selected.append(chunk)
            total_tokens += chunk_tokens
            chunk['token_count'] = chunk_tokens
        else:
            # Budget exhausted
            break
    
    print(f"✓ Selected {len(selected)} chunks ({total_tokens:,} tokens)")
    return selected


# --- STEP 5: GENERATE ANSWER WITH GEMINI ---

def generate_answer(
    query: str,
    context_chunks: List[Dict],
    model: str = LLM_MODEL,
    temperature: float = 0.3
) -> Dict:
    """
    Generate final answer using Gemini 2.5 Flash with retrieved context.
    
    Args:
        query: User's original query
        context_chunks: Selected context chunks
        model: Gemini model to use
        temperature: Sampling temperature (0-1, lower = more deterministic)
        
    Returns:
        Dictionary with answer text and metadata
    """
    # Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        metadata = json.loads(chunk['metadata_json'])
        title = metadata.get('title', 'Unknown')
        year = metadata.get('year', 'N/A')
        
        context_parts.append(
            f"[Source {i}] {title} ({year})\n{chunk['chunk_text']}\n"
        )
    
    context_text = "\n---\n".join(context_parts)
    
    # Construct prompt
    system_instruction = """You are a helpful movie information assistant. 
Answer the user's question based on the provided movie information sources. 
Be specific and cite which sources support your answer. 
If the sources don't contain enough information, acknowledge the limitations."""
    
    prompt = f"""Context Information:
{context_text}

User Question: {query}

Please provide a comprehensive answer based on the context above."""
    
    print(f"🤖 Generating answer with {model}...")
    
    try:
        # Count tokens in prompt
        prompt_tokens = count_tokens(system_instruction + prompt, model)
        print(f"  Prompt tokens: {prompt_tokens:,}")
        
        # Generate response
        response = genai_client.models.generate_content(
            model=model,
            contents=[prompt],
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=RESERVED_OUTPUT_TOKENS,
            )
        )
        
        answer_text = response.text
        
        # Extract usage metadata
        usage = response.usage_metadata
        result = {
            'answer': answer_text,
            'sources_used': len(context_chunks),
            'prompt_tokens': usage.prompt_token_count if usage else prompt_tokens,
            'output_tokens': usage.candidates_token_count if usage else len(answer_text) // 4,
            'total_tokens': usage.total_token_count if usage else None,
            'model': model
        }
        
        print(f"✓ Generated answer ({result['output_tokens']} tokens)")
        return result
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise

def embed_query(query_text: str) -> List[float]:
    try:
        response = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query_text]
        )
        
        if response.embeddings and len(response.embeddings) > 0:
            embedding = response.embeddings[0].values
            print(f"✓ Generated query embedding (dim={len(embedding)})")
            return embedding
        else:
            raise ValueError("No embedding returned from API")
            
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

def vector_search_bigquery(
    query_embedding: List[float],
    top_k: int = INITIAL_TOP_K,
    metadata_filter: Optional[str] = None
) -> List[Dict]:
    """
    Perform vector similarity search in BigQuery using VECTOR_SEARCH function.
    
    Args:
        query_embedding: Query vector to search for
        top_k: Number of results to return
        metadata_filter: Optional SQL WHERE clause for filtering (e.g., "year >= 2000")
        
    Returns:
        List of dictionaries containing chunk data and similarity scores
    """
    # Convert embedding list to SQL array format
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Base table query with optional filtering
    base_table_filter = f"WHERE {metadata_filter}" if metadata_filter else ""
    
    # Construct VECTOR_SEARCH query
    # This performs cosine similarity search on the embeddings
    query = f"""
    SELECT 
        base.movie_id,
        base.title,
        base.chunk_id,
        base.chunk_text,
        base.metadata_json,
        base.source_file,
        distance
    FROM VECTOR_SEARCH(
        (SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` {base_table_filter}),
        'embedding',
        (SELECT {embedding_str} AS embedding),
        distance_type => 'COSINE',
        top_k => {top_k}
    )
    """
    
    print(f"🔍 Executing BigQuery vector search (top_k={top_k})...")
    
    try:
        query_job = bq_client.query(query)
        results = list(query_job.result())
        
        # Convert to list of dictionaries
        search_results = []
        for row in results:
            search_results.append({
                'movie_id': row.movie_id,
                'title': row.title,
                'chunk_id': row.chunk_id,
                'chunk_text': row.chunk_text,
                'metadata_json': row.metadata_json,
                'source_file': row.source_file,
                'distance': float(row.distance)
            })
        
        print(f"✓ Retrieved {len(search_results)} candidate chunks")
        return search_results
        
    except Exception as e:
        print(f"Error in BigQuery vector search: {e}")
        raise

def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    top_k: int = RERANK_TOP_K
) -> List[Dict]:
    """
    Rerank candidates using a cross-encoder model for higher accuracy.
    
    Cross-encoders process query and document together, achieving better
    relevance scoring than simple vector similarity at the cost of speed.
    
    Args:
        query: Original user query
        candidates: List of candidate chunks from vector search
        top_k: Number of top results to return after reranking
        
    Returns:
        Reranked list of top_k candidates with rerank_score added
    """
    if not candidates:
        return []
    
    print(f"🔄 Reranking {len(candidates)} candidates with cross-encoder...")
    
    # Prepare query-document pairs for the cross-encoder
    # Truncate texts to avoid exceeding model's max length
    pairs = [
        (query[:200], candidate['chunk_text'][:400]) 
        for candidate in candidates
    ]
    
    try:
        # Get relevance scores from cross-encoder
        # Higher scores = more relevant
        scores = reranker.predict(pairs)
        
        # Add rerank scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)
        
        # Sort by rerank score (descending) and take top_k
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        
        print(f"✓ Reranked to top {len(reranked)} chunks")
        print(f"  Score range: [{reranked[-1]['rerank_score']:.2f}, {reranked[0]['rerank_score']:.2f}]")
        
        return reranked
        
    except Exception as e:
        print(f"Warning: Reranking failed ({e}), falling back to vector search ranking")
        return candidates[:top_k]
    
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# --- MAIN RAG PIPELINE ---

def rag_pipeline(
    user_query: str,
    initial_top_k: int = INITIAL_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    metadata_filter: Optional[str] = None,
    enable_reranking: bool = True,
    enable_deduplication: bool = True
) -> Dict:
    """
    Complete RAG pipeline: retrieve, rerank, and generate answer.
    
    Args:
        user_query: User's natural language question
        initial_top_k: Number of candidates to retrieve from BigQuery
        rerank_top_k: Number of chunks to keep after reranking
        max_context_tokens: Maximum tokens for context
        metadata_filter: Optional SQL filter (e.g., "year >= 2000")
        enable_reranking: Whether to use cross-encoder reranking
        enable_deduplication: Whether to deduplicate chunks by movie
        
    Returns:
        Dictionary containing answer and pipeline metadata
    """
    print(f"\n{'='*60}")
    print(f"RAG PIPELINE: {user_query[:80]}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # Step 1: Generate query embedding
        query_embedding = embed_query(user_query)
        
        # Step 2: Vector search in BigQuery
        candidates = vector_search_bigquery(
            query_embedding=query_embedding,
            top_k=initial_top_k,
            metadata_filter=metadata_filter
        )
        
        if not candidates:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources_used': 0,
                'error': 'No candidates found'
            }
        
        # Step 3: Reranking (optional but recommended)
        if enable_reranking:
            ranked_chunks = rerank_with_cross_encoder(
                query=user_query,
                candidates=candidates,
                top_k=rerank_top_k
            )
        else:
            ranked_chunks = candidates[:rerank_top_k]
        
        # Step 3b: Deduplication (optional)
        if enable_deduplication:
            ranked_chunks = deduplicate_chunks(ranked_chunks)
        
        # Step 4: Token-aware selection
        selected_chunks = select_chunks_within_token_budget(
            chunks=ranked_chunks,
            max_tokens=max_context_tokens
        )
        
        if not selected_chunks:
            return {
                'answer': "The relevant information exceeds token limits. Please refine your query.",
                'sources_used': 0,
                'error': 'Token budget exceeded'
            }
        
        # Step 5: Generate answer
        result = generate_answer(
            query=user_query,
            context_chunks=selected_chunks
        )
        
        # Add pipeline metadata
        result['pipeline_time_seconds'] = time.time() - start_time
        result['candidates_retrieved'] = len(candidates)
        result['chunks_after_reranking'] = len(ranked_chunks)
        result['query'] = user_query
        
        print(f"\n{'='*60}")
        print(f"✓ Pipeline completed in {result['pipeline_time_seconds']:.2f}s")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}\n")
        return {
            'answer': f"An error occurred: {str(e)}",
            'error': str(e),
            'pipeline_time_seconds': time.time() - start_time
        }

def print_detailed_results(result: Dict):
    """Pretty print RAG pipeline results (FIXED - no truncation)"""
    
    print("\n" + "="*70)
    print("RAG PIPELINE RESULTS")
    print("="*70)
    
    print(f"\n📝 QUERY:")
    print(f"   {result.get('query', 'N/A')}")
    
    print(f"\n📄 ANSWER:")
    answer = result.get('answer', 'N/A')
    # FIXED: Print FULL answer, no truncation
    print(f"   {answer}")
    
    print(f"\n📊 METRICS:")
    print(f"   Candidates Retrieved: {result.get('candidates_retrieved', 0)}")
    print(f"   After Reranking: {result.get('chunks_after_reranking', 0)}")
    print(f"   Sources Used: {result.get('sources_used', 0)}")
    print(f"   Reranker Used: {result.get('reranker_used', 'N/A')}")
    print(f"   Prompt Tokens: {result.get('prompt_tokens', 'N/A')}")
    print(f"   Output Tokens: {result.get('output_tokens', 'N/A')}")
    print(f"   Total Tokens: {result.get('total_tokens', 'N/A')}")
    print(f"   Pipeline Time: {result.get('pipeline_time_seconds', 0):.2f}s")
    
    if 'error' in result and result['error']:
        print(f"\n⚠️  ERROR: {result['error']}")
    
    print("\n" + "="*70 + "\n")