#!/usr/bin/env python3
"""Benchmark script comparing PostgreSQL and SQLite backends.

Usage:
    python scripts/benchmark_backends.py [--num-entities N] [--num-passages N]

This script compares performance of core operations:
- Entity insertion
- Passage insertion
- Vector search (entity and passage)
- Entity-passage linking
"""

import argparse
import os
import random
import tempfile
import time
from pathlib import Path

# Set up test environment
os.environ["REVERIE_CACHE_ENABLED"] = "false"


def generate_embedding(seed: int, dim: int = 1536) -> list[float]:
    """Generate a deterministic random embedding."""
    random.seed(seed)
    return [random.random() for _ in range(dim)]


def benchmark_entity_insert(db, num_entities: int) -> float:
    """Benchmark entity insertion."""
    start = time.perf_counter()
    for i in range(num_entities):
        db.upsert_entity(f"bench_entity_{i}", generate_embedding(i))
    db.commit()
    return time.perf_counter() - start


def benchmark_passage_insert(db, num_passages: int) -> float:
    """Benchmark passage insertion."""
    start = time.perf_counter()
    for i in range(num_passages):
        db.insert_passage(
            session_id=f"bench_session_{i // 10}",
            turn_index=i % 10,
            text=f"Benchmark passage {i} with some content",
            embedding=generate_embedding(i + 10000),
            user_text=f"User said {i}",
            user_embedding=generate_embedding(i + 20000),
            assistant_text=f"Assistant replied {i}",
            assistant_embedding=generate_embedding(i + 30000),
        )
    db.commit()
    return time.perf_counter() - start


def benchmark_entity_search(db, num_queries: int) -> float:
    """Benchmark entity vector search."""
    start = time.perf_counter()
    for i in range(num_queries):
        db.search_entities_by_embedding(generate_embedding(i + 100000), top_k=10)
    return time.perf_counter() - start


def benchmark_passage_search(db, num_queries: int) -> float:
    """Benchmark passage vector search."""
    start = time.perf_counter()
    for i in range(num_queries):
        db.search_passages_by_embedding(generate_embedding(i + 200000), top_k=10)
    return time.perf_counter() - start


def benchmark_entity_seeds(db, num_queries: int) -> float:
    """Benchmark entity seed finding."""
    start = time.perf_counter()
    for i in range(num_queries):
        db.find_entity_seeds(generate_embedding(i + 300000), tau=0.5)
    return time.perf_counter() - start


def benchmark_linking(db, num_links: int) -> float:
    """Benchmark entity-passage linking."""
    # Get entity and passage IDs
    entities = db.list_entities(limit=100)[0]
    passages = db.list_passages(limit=100)[0]

    if not entities or not passages:
        return 0.0

    start = time.perf_counter()
    for i in range(min(num_links, len(entities) * len(passages))):
        entity = entities[i % len(entities)]
        passage = passages[i % len(passages)]
        db.link_entity_passage(entity["id"], passage["id"])
    db.commit()
    return time.perf_counter() - start


def run_benchmark(backend: str, db_path: str | None, args):
    """Run benchmark for a specific backend."""
    from reverie.db.factory import create_db

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {backend.upper()} backend")
    print(f"{'=' * 60}")

    if backend == "sqlite":
        db = create_db("sqlite", db_path=db_path, run_migrations=True)
    else:
        # PostgreSQL - use test database
        from dotenv import load_dotenv
        load_dotenv()

        # Set required env vars for settings validation
        os.environ["REVERIE_DB_NAME"] = "reverie_test"
        os.environ["REVERIE_CORS_ORIGIN"] = "*"
        os.environ["REVERIE_OPENIE_MODEL"] = "gpt-4.1-mini"
        os.environ["REVERIE_RECOGNITION_MODEL"] = "gpt-4.1-mini"
        os.environ["REVERIE_PREPROCESSOR_MODEL"] = "gpt-4.1-mini"

        from reverie.config import get_settings
        settings = get_settings()
        db = create_db(
            "postgres",
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            dbname=settings.db_name,
        )

    results = {}

    try:
        # Entity insertion
        print(f"\nInserting {args.num_entities} entities...")
        results["entity_insert"] = benchmark_entity_insert(db, args.num_entities)
        print(f"  Time: {results['entity_insert']:.3f}s")
        print(f"  Rate: {args.num_entities / results['entity_insert']:.1f} entities/s")

        # Passage insertion
        print(f"\nInserting {args.num_passages} passages...")
        results["passage_insert"] = benchmark_passage_insert(db, args.num_passages)
        print(f"  Time: {results['passage_insert']:.3f}s")
        print(f"  Rate: {args.num_passages / results['passage_insert']:.1f} passages/s")

        # Entity search
        print(f"\nRunning {args.num_queries} entity searches...")
        results["entity_search"] = benchmark_entity_search(db, args.num_queries)
        print(f"  Time: {results['entity_search']:.3f}s")
        print(f"  Rate: {args.num_queries / results['entity_search']:.1f} queries/s")

        # Passage search
        print(f"\nRunning {args.num_queries} passage searches...")
        results["passage_search"] = benchmark_passage_search(db, args.num_queries)
        print(f"  Time: {results['passage_search']:.3f}s")
        print(f"  Rate: {args.num_queries / results['passage_search']:.1f} queries/s")

        # Entity seeds
        print(f"\nRunning {args.num_queries} entity seed searches...")
        results["entity_seeds"] = benchmark_entity_seeds(db, args.num_queries)
        print(f"  Time: {results['entity_seeds']:.3f}s")
        print(f"  Rate: {args.num_queries / results['entity_seeds']:.1f} queries/s")

        # Linking
        print(f"\nCreating {args.num_links} entity-passage links...")
        results["linking"] = benchmark_linking(db, args.num_links)
        print(f"  Time: {results['linking']:.3f}s")
        print(f"  Rate: {args.num_links / max(results['linking'], 0.001):.1f} links/s")

        # Statistics
        stats = db.get_stats()
        print(f"\nDatabase stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    finally:
        db.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PostgreSQL vs SQLite backends")
    parser.add_argument("--num-entities", type=int, default=100, help="Number of entities to insert")
    parser.add_argument("--num-passages", type=int, default=100, help="Number of passages to insert")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of search queries")
    parser.add_argument("--num-links", type=int, default=100, help="Number of links to create")
    parser.add_argument("--sqlite-only", action="store_true", help="Only benchmark SQLite")
    parser.add_argument("--postgres-only", action="store_true", help="Only benchmark PostgreSQL")
    args = parser.parse_args()

    results = {}

    # SQLite benchmark
    if not args.postgres_only:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            results["sqlite"] = run_benchmark("sqlite", db_path, args)
        finally:
            Path(db_path).unlink(missing_ok=True)

    # PostgreSQL benchmark
    if not args.sqlite_only:
        try:
            results["postgres"] = run_benchmark("postgres", None, args)
        except Exception as e:
            print(f"\nPostgreSQL benchmark failed: {e}")
            print("Make sure PostgreSQL is running and reverie_test database exists")

    # Comparison
    if "sqlite" in results and "postgres" in results:
        print(f"\n{'=' * 60}")
        print("COMPARISON (PostgreSQL vs SQLite)")
        print(f"{'=' * 60}")

        for op in results["sqlite"]:
            pg_time = results["postgres"][op]
            sq_time = results["sqlite"][op]
            if sq_time > 0:
                ratio = pg_time / sq_time
                faster = "PostgreSQL" if ratio < 1 else "SQLite"
                factor = 1/ratio if ratio < 1 else ratio
                print(f"{op:20s}: PG={pg_time:.3f}s, SQ={sq_time:.3f}s ({faster} {factor:.1f}x faster)")


if __name__ == "__main__":
    main()
