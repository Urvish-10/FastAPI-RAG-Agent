"""
app/evaluation/setup_eval_data.py

Provisions the evaluation fixture data for RAG evals:
  1. Registers (or logs in) a dedicated eval user, isolated from real user data.
  2. Uploads each fixture PDF in app/evaluation/fixtures/ through the app's own
     /documents/upload endpoint (so ingestion is tested end-to-end, same as prod).
  3. Polls each document until its persisted ingestion status is ready.
  4. Writes a provisioned-document manifest for the RAGAS runner.
  5. Writes app/evaluation/eval_context.json — the manifest the RAGAS runner
     reads to know which user/session/document IDs to use.

Run this once whenever fixtures change (new/updated PDFs), not on every eval run.

Usage:
    python -m app.evaluation.setup_eval_data
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

# ---- Config -----------------------------------------------------------
BASE_URL = "http://localhost:8000/api/v1"
EVAL_EMAIL = "testevals@yopmail.com"
EVAL_PASSWORD = "Test105*"  # noqa: S105 — eval-only fixture account

THIS_DIR = Path(__file__).parent
FIXTURES_DIR = THIS_DIR / "fixtures"
OUTPUT_MANIFEST = THIS_DIR / "eval_context.json"

POLL_INTERVAL_SECONDS = 2
# The handbook can span multiple rate-limited embedding batches. Keep this
# comfortably above the worker's inter-batch pauses on a first ingestion.
POLL_TIMEOUT_SECONDS = 900
# ------------------------------------------------------------------------


async def ensure_eval_user(client: httpx.AsyncClient) -> str:
    """Register the eval user if needed, then log in. Returns access token."""
    register_resp = await client.post(
        "/auth/register",
        json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD},
    )
    if register_resp.status_code not in (200, 201, 400, 409):
        register_resp.raise_for_status()
    if register_resp.status_code in (200, 201):
        print(f"[setup] Registered eval user: {EVAL_EMAIL}")
    else:
        print(f"[setup] Eval user already exists: {EVAL_EMAIL}")

    login_resp = await client.post(
        "/auth/login",
        json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD},
    )
    login_resp.raise_for_status()
    token = login_resp.json()["access_token"]
    print("[setup] Logged in as eval user")
    return token


async def upload_fixture(
    client: httpx.AsyncClient, headers: dict, pdf_path: Path
) -> str:
    """Upload one fixture PDF, poll until its persisted status is ready.

    The task-status endpoint exposes Celery states (for example ``SUCCESS``),
    while the document endpoint exposes the application states ``ready`` and
    ``failed`` that determine whether the fixture is usable for evaluation.
    """
    with pdf_path.open("rb") as f:
        upload_resp = await client.post(
            "/documents/upload",
            headers=headers,
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
    upload_resp.raise_for_status()
    document_id = upload_resp.json()["id"]
    print(f"[setup] Uploaded {pdf_path.name} -> document_id={document_id}, polling ingestion...")

    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        document_resp = await client.get(f"/documents/{document_id}", headers=headers)
        document_resp.raise_for_status()
        document = document_resp.json()
        status = document.get("status")
        if status == "ready":
            print(f"[setup]   {pdf_path.name} ready (document_id={document_id})")
            return document_id
        if status == "failed":
            raise RuntimeError(
                f"Ingestion failed for {pdf_path.name}: "
                f"{document.get('error_message', 'no error message returned')}"
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    raise TimeoutError(f"Ingestion for {pdf_path.name} did not finish within {POLL_TIMEOUT_SECONDS}s")


async def main() -> None:
    if not FIXTURES_DIR.exists():
        print(f"[setup] ERROR: fixtures directory not found at {FIXTURES_DIR}")
        sys.exit(1)

    pdf_paths = sorted(FIXTURES_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"[setup] ERROR: no PDFs found in {FIXTURES_DIR}")
        sys.exit(1)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        token = await ensure_eval_user(client)
        headers = {"Authorization": f"Bearer {token}"}

        documents: dict[str, str] = {}
        for pdf_path in pdf_paths:
            document_id = await upload_fixture(client, headers, pdf_path)
            documents[pdf_path.name] = document_id

    manifest = {
        "eval_email": EVAL_EMAIL,
        "documents": documents,  # filename -> document_id
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[setup] Wrote manifest to {OUTPUT_MANIFEST}")
    print("[setup] Done. You can now run: python -m app.evaluation.ragas_runner")


if __name__ == "__main__":
    asyncio.run(main())
