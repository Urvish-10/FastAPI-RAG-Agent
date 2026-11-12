"""
Ingest the PDFs in the repo-level docs/ folder directly into Pinecone (no API server / Celery).

Uses the exact same chunking + embedding + upsert code as the production worker,
but writes to a dedicated namespace (settings.EVAL_NAMESPACE) so real users'
data is never touched. Vector IDs are deterministic, so re-running is idempotent.

Usage:
    uv run python -m app.evaluation.ingest            # skip if namespace already populated
    uv run python -m app.evaluation.ingest --force    # re-ingest everything
"""
import argparse
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.services.pdf_processor import load_and_chunk_pdf
from app.services.vector_store import get_pinecone_client, upsert_chunks

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def eval_document_id(pdf_path: Path) -> str:
    """Stable document id derived from the filename."""
    return f"eval-{pdf_path.stem}"


def namespace_vector_count(namespace: str) -> int:
    pc = get_pinecone_client()
    stats = pc.Index(settings.PINECONE_INDEX_NAME).describe_index_stats()
    ns = (stats.namespaces or {}).get(namespace)
    return ns.vector_count if ns else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--force", action="store_true", help="re-ingest even if populated")
    args = parser.parse_args()

    namespace = settings.EVAL_NAMESPACE
    existing = namespace_vector_count(namespace)
    if existing and not args.force:
        print(f"Namespace '{namespace}' already has {existing} vectors. Use --force to re-ingest.")
        return

    pdfs = sorted(args.docs_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.docs_dir}")

    for pdf in pdfs:
        doc_id = eval_document_id(pdf)
        chunks, pages = load_and_chunk_pdf(
            file_path=str(pdf),
            document_id=doc_id,
            user_id=namespace,
            original_filename=pdf.name,
        )
        print(f"{pdf.name}: {pages} pages -> {len(chunks)} chunks. Embedding (rate-limited)...")
        n = upsert_chunks(chunks=chunks, namespace=namespace)
        print(f"  upserted {n} vectors into namespace '{namespace}' (document_id={doc_id})")


if __name__ == "__main__":
    main()
