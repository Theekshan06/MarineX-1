#!/usr/bin/env python3
"""
Working Enhanced RAG System - Uses fast local models for immediate testing
"""

import os
import json
import re
import duckdb
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import chromadb
import groq
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv

@dataclass 
class QueryResult:
    enhanced_sql: str
    method: str
    similarity: float
    execution_time: float
    metadata: Dict[str, Any]

class WorkingChromaManager:
    """ChromaDB manager using fast local model (all-MiniLM-L6-v2)"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token
        self.current_model = "all-MiniLM-L6-v2"  # Local-only model, no API needed
        self._init_fast_model()
        self._init_chromadb()
    
    def _init_fast_model(self):
        """Initialize fast, small embedding model"""
        try:
            print(f"[INFO] Loading fast embedding model: {self.current_model}")
            self.embedding_model = SentenceTransformer(self.current_model)
            print(f"[SUCCESS] Loaded fast model (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def _init_chromadb(self):
        """Initialize ChromaDB with fast embedding function"""
        self.client = chromadb.PersistentClient(path="./working_enhanced_chroma_db")
        self.collection_name = "working_optimized_argo_queries"
        
        # Fast embedding function with ChromaDB compatibility
        class FastEmbeddingFunction:
            def __init__(self, model):
                self.model = model
            
            def name(self):
                return "fast_sentence_transformers"
            
            def __call__(self, input):
                # Handle ChromaDB interface (single string or list)
                if isinstance(input, str):
                    input = [input]
                
                embeddings = self.model.encode(input, convert_to_tensor=False)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                return embeddings.tolist()
        
        self.embedding_function = FastEmbeddingFunction(self.embedding_model)
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"[INFO] Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Working optimized ARGO queries with fast embeddings"}
            )
            print(f"[INFO] Created new collection: {self.collection_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.current_model,
            'embedding_dim': self.embedding_model.get_sentence_embedding_dimension(),
            'api_based': False,
            'embedding_method': 'sentence_transformers_local'
        }
    
    def populate_with_hybrid_data(self, force_reload=True):
        """Load hybrid data (original + enhanced analytical queries)"""

        print("[INFO] Using HYBRID ChromaDB data (407 simple + 6 enhanced analytical queries)")

        # Always force reload to use hybrid data
        if force_reload:
            print("[INFO] Force reloading with hybrid data...")
        else:
            # Check if collection already has data
            try:
                current_count = self.collection.count()
                if current_count > 0:
                    print(f"[INFO] ChromaDB already has {current_count} queries - skipping reload")
                    print(f"[INFO] Use force_reload=True to upgrade to hybrid data")
                    return
            except Exception as e:
                print(f"[INFO] Could not check collection count: {e}")

        # Load comprehensive FloatChat data (original + enhanced + ARGO specific)
        floatchat_file = 'optimized_chromadb_data_floatchat_final.json'
        try:
            with open(floatchat_file, 'r', encoding='utf-8') as f:
                hybrid_data = json.load(f)
            print(f"[SUCCESS] Loaded comprehensive FloatChat data file: {floatchat_file}")
        except FileNotFoundError:
            # Fallback to previous hybrid file
            print(f"[WARNING] {floatchat_file} not found! Trying hybrid file...")
            try:
                with open('optimized_chromadb_data_hybrid_fixed.json', 'r', encoding='utf-8') as f:
                    hybrid_data = json.load(f)
                print("[INFO] Using previous hybrid file (missing ARGO-specific queries)")
            except FileNotFoundError:
                print("[WARNING] No comprehensive data found! Falling back to basic data...")
                try:
                    with open('optimized_chromadb_data.json', 'r', encoding='utf-8') as f:
                        hybrid_data = json.load(f)
                except FileNotFoundError:
                    print("[ERROR] No ChromaDB data file found!")
                    print("Please ensure optimized_chromadb_data_floatchat_final.json exists")
                    return
        
        queries = hybrid_data['queries']
        total_queries = len(queries)
        collection_info = hybrid_data.get('collection_info', {})

        print(f"[INFO] Recreating ChromaDB with {total_queries} comprehensive FloatChat queries...")

        # Print breakdown if available
        if collection_info.get('query_breakdown'):
            breakdown = collection_info['query_breakdown']
            print(f"[INFO] Query breakdown: {breakdown.get('simple_column_access', 0)} simple + {breakdown.get('enhanced_analytical', 0)} enhanced + {breakdown.get('argo_floatchat_specific', 0)} ARGO-specific")
        print(f"[INFO] Collection type: {collection_info.get('name', 'hybrid_argo_queries')}")

        # Get query distribution info if available
        original_count = collection_info.get('original_queries', 0)
        enhanced_count = collection_info.get('enhanced_queries', 0)
        if original_count and enhanced_count:
            print(f"[INFO] Queries breakdown: {original_count} simple + {enhanced_count} enhanced analytical")

        # FORCE DELETE existing ChromaDB collection to start fresh
        try:
            self.client.delete_collection(self.collection_name)
            print(f"[INFO] DELETED existing collection: {self.collection_name}")
            time.sleep(1)  # Wait for deletion to complete
        except Exception as e:
            print(f"[INFO] No existing collection to delete: {e}")

        # CREATE FRESH collection with hybrid data
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "Hybrid ARGO queries: simple column access + enhanced analytical templates",
                    "version": "hybrid_v1.0",
                    "total_queries": total_queries,
                    "original_queries": original_count,
                    "enhanced_queries": enhanced_count,
                    "upgrade_date": datetime.now().isoformat()
                }
            )
            print(f"[SUCCESS] Created fresh hybrid collection: {self.collection_name}")
        except Exception as e:
            print(f"[ERROR] Failed to create collection: {e}")
            return

        # Process in reasonable batches
        batch_size = 50
        total_batches = (total_queries - 1) // batch_size + 1

        print(f"[INFO] Processing {total_queries} queries in {total_batches} batches...")

        for i in range(0, total_queries, batch_size):
            batch = queries[i:i+batch_size]
            batch_num = i // batch_size + 1

            print(f"[INFO] Processing batch {batch_num}/{total_batches} ({len(batch)} queries)...")

            try:
                ids = [q['id'] for q in batch]
                documents = [q['content'] for q in batch]
                metadatas = [q['metadata'] for q in batch]

                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"[SUCCESS] Batch {batch_num} added successfully")

            except Exception as e:
                print(f"[ERROR] Failed to add batch {batch_num}: {e}")
                continue

        # Verify final count
        try:
            final_count = self.collection.count()
            print(f"[SUCCESS] ChromaDB upgrade complete!")
            print(f"[SUCCESS] Total queries loaded: {final_count}/{total_queries}")

            if original_count and enhanced_count:
                print(f"[SUCCESS] Now supports both simple queries AND enhanced analytical queries!")
                print(f"[SUCCESS] Simple column access: {original_count} queries")
                print(f"[SUCCESS] Complex analytics: {enhanced_count} enhanced templates")

        except Exception as e:
            print(f"[WARNING] Could not verify final count: {e}")

        print(f"[SUCCESS] Hybrid ChromaDB ready for enhanced query processing!")
    
    def classify_query_intent(self, query_text: str) -> Dict[str, Any]:
        """Classify query intent for better context-aware matching"""
        query_lower = query_text.lower()
        
        # Intent classification patterns
        intent_patterns = {
            'individual_profile': [
                'each profile', 'per profile', 'individual profile', 'profile by profile',
                'for each profile', 'every profile', 'profile-specific', 'profile level'
            ],
            'individual_float': [
                'each float', 'per float', 'individual float', 'float by float', 
                'for each float', 'every float', 'float-specific', 'float level'
            ],
            'geographic': [
                'latitude', 'longitude', 'region', 'area', 'basin', 'geographic',
                'location', 'spatial', 'by latitude', 'by region', 'geographic distribution'
            ],
            'temporal': [
                'time', 'date', 'temporal', 'seasonal', 'monthly', 'yearly',
                'over time', 'by date', 'chronological', 'time series'
            ],
            'global_aggregate': [
                'overall', 'total', 'all profiles', 'all floats', 'across all',
                'global', 'entire dataset', 'complete', 'comprehensive'
            ],
            'simple_retrieval': [
                'get', 'show', 'retrieve', 'display', 'list', 'fetch',
                'give me', 'show me', 'data', 'values'
            ]
        }
        
        # Parameter detection
        parameter_patterns = {
            'temperature': ['temp', 'temperature', 'thermal', 'warm', 'cold', 'heat'],
            'salinity': ['sal', 'salinity', 'salt', 'salty', 'fresh', 'brackish'],
            'pressure': ['pressure', 'depth', 'deep', 'shallow', 'dbar'],
            'comprehensive': ['all', 'complete', 'full', 'comprehensive', 'statistics']
        }
        
        # Statistical operation detection  
        operation_patterns = {
            'average': ['average', 'avg', 'mean'],
            'count': ['count', 'number', 'how many'],
            'min_max': ['min', 'max', 'minimum', 'maximum', 'highest', 'lowest'],
            'statistics': ['stats', 'statistics', 'analysis', 'summary']
        }
        
        # Classify intent
        detected_intent = 'unknown'
        intent_confidence = 0.0
        
        for intent, patterns in intent_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in query_lower)
            if matches > 0:
                confidence = matches / len(patterns)
                if confidence > intent_confidence:
                    detected_intent = intent
                    intent_confidence = confidence
        
        # Detect parameters
        detected_parameters = []
        for param, patterns in parameter_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_parameters.append(param)
        
        # Detect operations
        detected_operations = []
        for op, patterns in operation_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_operations.append(op)
        
        return {
            'intent': detected_intent,
            'confidence': intent_confidence,
            'parameters': detected_parameters,
            'operations': detected_operations,
            'grouping_level': self._infer_grouping_level(detected_intent)
        }
    
    def _infer_grouping_level(self, intent: str) -> str:
        """Infer grouping level from intent"""
        grouping_map = {
            'individual_profile': 'profile',
            'individual_float': 'float', 
            'geographic': 'region',
            'temporal': 'time',
            'global_aggregate': 'global',
            'simple_retrieval': 'none'
        }
        return grouping_map.get(intent, 'unknown')
    
    def preprocess_query(self, query_text: str) -> str:
        """Preprocess query for better semantic matching"""
        # Oceanographic term expansions
        expansions = {
            'temp': 'temperature thermal ocean',
            'sal': 'salinity salt seawater',
            'deep': 'depth pressure abyssal',
            'float': 'ARGO float CTD instrument',
            'warm': 'temperature thermal hot',
            'cold': 'temperature thermal cool',
            'salty': 'salinity salt concentration',
            'fresh': 'salinity freshwater low salt',
            'profile': 'oceanographic profile measurement',
            'anomaly': 'unusual abnormal outlier',
            'water': 'seawater ocean marine'
        }
        
        # Expand query with related terms
        expanded_query = query_text.lower()
        for term, expansion in expansions.items():
            if term in expanded_query:
                expanded_query += f" {expansion}"
        
        return expanded_query

    def context_aware_similarity_scoring(self, query_intent: Dict, results: List[Dict]) -> List[Dict]:
        """Apply context-aware similarity scoring based on intent matching"""
        for result in results:
            base_similarity = result['similarity']
            metadata = result.get('metadata', {})
            
            # Context penalties and bonuses
            context_score = 1.0
            
            # Grouping level matching bonus/penalty
            query_grouping = query_intent.get('grouping_level', 'unknown')
            result_grouping = metadata.get('grouping_level', 'unknown')
            
            if query_grouping != 'unknown' and result_grouping != 'unknown':
                if query_grouping == result_grouping:
                    context_score *= 1.3  # 30% bonus for matching grouping level
                else:
                    context_score *= 0.6   # 40% penalty for wrong grouping level
            
            # Intent matching bonus
            query_intent_type = query_intent.get('intent', 'unknown')
            result_intent = metadata.get('intent', 'unknown')
            
            if query_intent_type != 'unknown' and result_intent != 'unknown':
                if query_intent_type == result_intent:
                    context_score *= 1.2  # 20% bonus for matching intent
                elif self._intent_conflict(query_intent_type, result_intent):
                    context_score *= 0.5   # 50% penalty for conflicting intent
            
            # Parameter matching bonus
            query_params = set(query_intent.get('parameters', []))
            result_param = metadata.get('parameter', '')
            
            if result_param in query_params:
                context_score *= 1.15  # 15% bonus for parameter match
            
            # Apply context-aware scoring
            result['context_aware_similarity'] = min(1.0, base_similarity * context_score)
            result['context_score'] = context_score
            result['query_intent'] = query_intent
        
        return results
    
    def _intent_conflict(self, intent1: str, intent2: str) -> bool:
        """Check if two intents are conflicting"""
        conflicts = [
            ('individual_profile', 'geographic'),
            ('individual_float', 'geographic'),
            ('individual_profile', 'global_aggregate'),
            ('individual_float', 'global_aggregate'),
            ('simple_retrieval', 'global_aggregate')
        ]
        
        return (intent1, intent2) in conflicts or (intent2, intent1) in conflicts
    
    def semantic_search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Perform multi-stage semantic search with context awareness"""
        try:
            # Stage 1: Classify query intent
            query_intent = self.classify_query_intent(query_text)
            
            # Stage 2: Standard semantic search
            processed_query = self.preprocess_query(query_text)
            
            # Get more results for filtering
            raw_results = self.collection.query(
                query_texts=[processed_query],
                n_results=min(top_k * 3, 30),  # Get 3x results for filtering
                include=['documents', 'metadatas', 'distances']
            )
            
            if not raw_results['documents'][0]:
                return []
            
            # Convert to structured results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                raw_results['documents'][0],
                raw_results['metadatas'][0], 
                raw_results['distances'][0]
            )):
                similarity = max(0.0, 1.0 - distance)
                
                search_results.append({
                    'document': doc,
                    'metadata': metadata,
                    'similarity': similarity,
                    'rank': i + 1
                })
            
            # Stage 3: Apply context-aware scoring
            context_scored_results = self.context_aware_similarity_scoring(query_intent, search_results)
            
            # Stage 4: Re-rank by context-aware similarity
            context_scored_results.sort(key=lambda x: x['context_aware_similarity'], reverse=True)
            
            # Return top_k results
            return context_scored_results[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Multi-stage semantic search failed: {e}")
            return []

class WorkingRAGSystem:
    """Working RAG system with fast local embeddings"""
    
    def __init__(self, groq_api_key: str, hf_token: Optional[str] = None):
        print("[INFO] Initializing Working Enhanced RAG System...")
        
        # Initialize components
        self.chroma_manager = WorkingChromaManager(hf_token)
        self.query_engine = self._init_query_engine()
        
        # Initialize Groq client
        self.groq_client = groq.Groq(api_key=groq_api_key)
        
        print("[SUCCESS] Working RAG System ready!")
    
    def _init_query_engine(self):
        """Initialize DuckDB query engine"""
        conn = duckdb.connect()
        parquet_path = "./parquet_data"
        
        tables = {
            'floats': f"{parquet_path}/floats.parquet",
            'profiles': f"{parquet_path}/profiles.parquet",
            'measurements': f"{parquet_path}/measurements.parquet"
        }
        
        for table_name, file_path in tables.items():
            if os.path.exists(file_path):
                try:
                    conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS 
                    SELECT * FROM read_parquet('{file_path}')
                    """)
                    print(f"[INFO] Setup table: {table_name}")
                except Exception as e:
                    print(f"[WARNING] Failed to setup {table_name}: {e}")
        
        return conn
    
    def setup_system(self, force_reload=True):
        """Setup the system with hybrid data"""
        print("[INFO] Setting up ChromaDB with HYBRID data (original + enhanced analytical)...")
        self.chroma_manager.populate_with_hybrid_data(force_reload=force_reload)
        print("[SUCCESS] Hybrid RAG system setup complete!")
    
    def generate_sql(self, user_query: str, rag_context: List[Dict]) -> str:
        """Generate SQL using LLM"""
        context_parts = []
        for result in rag_context[:3]:
            context_parts.append(f"Context: {result['document']}")
            context_parts.append(f"Similarity: {result['similarity']:.3f}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are an expert ARGO oceanographic database SQL generator.

DATABASE SCHEMA (DuckDB/Parquet):
- floats: float_id, wmo_number, current_status, deployment_date, deployment_latitude, deployment_longitude
- profiles: profile_id, float_id, profile_date, latitude, longitude, max_pressure
- measurements: measurement_id, profile_id, pressure, temperature, salinity, temperature_qc, salinity_qc

RULES:
1. Use exact column names from schema
2. Quality filters: temperature_qc <= 2, salinity_qc <= 2 for good data
3. Join pattern: FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id
4. NO LIMIT unless specifically requested
5. Return ONLY SQL, no explanations"""
        
        user_prompt = f"""Generate SQL for: "{user_query}"

Context:
{context}

Return only the SQL query."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            sql = response.choices[0].message.content.strip()
            sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\s*```\s*$', '', sql)
            
            return sql.strip()
            
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            return ""
    
    def process_query(self, user_query: str) -> QueryResult:
        """Process query with optimized similarity"""
        start_time = datetime.now()
        
        # Semantic search
        rag_results = self.chroma_manager.semantic_search(user_query, top_k=5)
        
        # Use context-aware similarity instead of raw similarity
        max_similarity = rag_results[0]['context_aware_similarity'] if rag_results else 0.0
        base_similarity = rag_results[0]['similarity'] if rag_results else 0.0
        context_score = rag_results[0]['context_score'] if rag_results else 1.0
        rag_sql = ""
        
        if rag_results:
            best_doc = rag_results[0]['document']
            sql_patterns = [
                r'SQL Query:\s*(SELECT.*?)(?:\n\n|\nUsage|\nExpected|\nQuery Variations|$)',
                r'SQL:\s*(SELECT.*?)(?:\n\n|\nUsage|\nExpected|\nQuery Variations|$)',
                r'(SELECT.*?)(?:\n\n|\nUsage|\nExpected|\nQuery Variations|$)'
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, best_doc, re.IGNORECASE | re.DOTALL)
                if match:
                    rag_sql = match.group(1).strip().rstrip(';')
                    # Clean up any remaining explanatory text
                    rag_sql = re.sub(r'\s+(Usage:|Expected Results:|Query Variations:).*$', '', rag_sql, flags=re.IGNORECASE | re.DOTALL)
                    break
        
        # Context-aware similarity thresholds
        if max_similarity >= 0.40 and rag_sql:  # Higher threshold for context-aware
            final_sql = rag_sql
            method = "rag_direct_context_match"
        elif max_similarity >= 0.25:  # Medium threshold with context consideration
            llm_sql = self.generate_sql(user_query, rag_results)
            final_sql = llm_sql if llm_sql else rag_sql
            method = "llm_enhanced_context_aware"
        else:
            llm_sql = self.generate_sql(user_query, rag_results)
            final_sql = llm_sql
            method = "llm_generated_low_context"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResult(
            enhanced_sql=final_sql,
            method=method,
            similarity=max_similarity,
            execution_time=execution_time,
            metadata={
                'rag_results_count': len(rag_results),
                'embedding_model': self.chroma_manager.get_model_info(),
                'base_similarity': base_similarity,
                'context_score': context_score,
                'query_intent': rag_results[0].get('query_intent', {}) if rag_results else {}
            }
        )
    
    def execute_query(self, sql: str) -> Tuple[List[Dict], bool]:
        """Execute SQL query"""
        try:
            sql = sql.strip().rstrip(';')
            result = self.query_engine.execute(sql).fetchall()
            columns = [desc[0] for desc in self.query_engine.description]
            data = [dict(zip(columns, row)) for row in result]
            return data, True
        except Exception as e:
            print(f"[ERROR] Query execution failed: {e}")
            return [], False
    
    def test_and_execute(self, user_query: str, show_results: int = 5):
        """Test query and show results"""
        print(f"\n[QUERY] {user_query}")
        print("=" * 60)
        
        result = self.process_query(user_query)
        
        print(f"[METHOD] {result.method}")
        print(f"[CONTEXT_SIMILARITY] {result.similarity:.4f}")
        print(f"[BASE_SIMILARITY] {result.metadata.get('base_similarity', 0):.4f}")
        print(f"[CONTEXT_SCORE] {result.metadata.get('context_score', 1.0):.2f}x")
        print(f"[INTENT] {result.metadata.get('query_intent', {}).get('intent', 'unknown')}")
        print(f"[GROUPING] {result.metadata.get('query_intent', {}).get('grouping_level', 'unknown')}")
        print(f"[TIME] {result.execution_time:.3f}s")
        print(f"[SQL] {result.enhanced_sql}")
        
        if result.enhanced_sql:
            data, success = self.execute_query(result.enhanced_sql)
            
            if success and data:
                print(f"\n[SUCCESS] Retrieved {len(data)} records")
                for i, row in enumerate(data[:show_results]):
                    print(f"  {i+1}: {row}")
                if len(data) > show_results:
                    print(f"  ... and {len(data) - show_results} more")
            elif success:
                print("\n[INFO] Query executed but no data returned")
            else:
                print("\n[ERROR] Query execution failed")
        else:
            print("\n[ERROR] No SQL generated")
        
        return result

def main():
    """Test the working RAG system"""
    
    load_dotenv()
    # Your API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    print("Working Enhanced RAG System Test")
    print("Using Fast Local Model + Optimized ChromaDB")
    print("=" * 60)
    
    try:
        # Initialize system
        rag_system = WorkingRAGSystem(GROQ_API_KEY, HF_TOKEN)
        
        # Setup system
        rag_system.setup_system()
        
        # Test queries to show similarity improvements
        test_queries = [
            "float_id",              # Should now get better similarity
            "get temperature",       # Should work well  
            "count floats",          # Should match well
            "salinity data",         # Should get good match
            "show profiles"          # Should work well
        ]
        
        print("\n" + "=" * 80)
        print("TESTING ENHANCED SEMANTIC SIMILARITY (FAST VERSION)")
        print("=" * 80)
        
        results = []
        for query in test_queries:
            result = rag_system.test_and_execute(query, show_results=3)
            results.append({
                'query': query,
                'similarity': result.similarity,
                'method': result.method
            })
            print("\n" + "-" * 60)
        
        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY") 
        print("=" * 60)
        
        high_sim = [r for r in results if r['similarity'] >= 0.4]
        print(f"High Similarity (>=0.4): {len(high_sim)}/{len(results)}")
        
        for r in results:
            status = "HIGH" if r['similarity'] >= 0.4 else "MEDIUM" if r['similarity'] >= 0.25 else "LOW"
            print(f"  '{r['query']}': {r['similarity']:.3f} ({status})")
        
        print(f"\nFast local embeddings working with optimized ChromaDB!")
        print(f"Ready to upgrade to API-based system when HF permissions are fixed.")
        
    except Exception as e:
        print(f"[ERROR] System failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
