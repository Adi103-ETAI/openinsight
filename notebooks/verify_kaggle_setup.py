"""Verify that all prerequisites for Kaggle ingestion are met.

Run this BEFORE the main ingestion notebook to catch configuration errors
early (no point starting a 6-hour GPU run if MongoDB is unreachable).

Usage on Kaggle:
    python notebooks/verify_kaggle_setup.py

Exit code 0 = all checks pass, ready to ingest
Exit code 1 = some check failed — fix before running ingestion
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print check result + return ok flag."""
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: {detail}")
    return ok


def main() -> int:
    print("=" * 60)
    print("OpenInsight — Kaggle Setup Verification")
    print("=" * 60)
    print()

    all_ok = True

    # --- 1. Environment ---
    print("1. Environment")
    is_kaggle = os.path.exists("/kaggle")
    check(
        "Kaggle environment",
        is_kaggle,
        "running on Kaggle" if is_kaggle else "NOT on Kaggle (running locally — some checks will skip)",
    )

    # --- 2. GPU ---
    print("\n2. GPU")
    try:
        subprocess.check_call([sys.executable, "-c", "import torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            all_ok &= check("CUDA available", True, f"{gpu_name} ({vram:.1f} GB VRAM)")
        else:
            all_ok &= check("CUDA available", False, "No GPU — enable GPU T4 in Kaggle settings")
    except Exception as e:
        all_ok &= check("torch installed", False, str(e)[:80])

    # --- 3. Kaggle Secrets ---
    print("\n3. Kaggle Secrets")
    secrets_ok = True
    try:
        from kaggle_secrets import UserSecretsClient
        sc = UserSecretsClient()
        for name in ["MONGODB_URL", "ZILLIZ_URI", "ZILLIZ_TOKEN"]:
            try:
                val = sc.get_secret(name)
                if val:
                    check(f"Secret '{name}'", True, f"set ({len(val)} chars)")
                else:
                    check(f"Secret '{name}'", False, "empty")
                    secrets_ok = False
            except Exception:
                check(f"Secret '{name}'", False, "not found — add it in Add-ons → Secrets")
                secrets_ok = False
        # NCBI API key is optional
        try:
            ncbi = sc.get_secret("NCBI_API_KEY")
            check("Secret 'NCBI_API_KEY'", bool(ncbi), "set (10 req/sec)" if ncbi else "not set (3 req/sec rate limit)")
        except Exception:
            check("Secret 'NCBI_API_KEY'", False, "not set (optional but recommended)")
    except ImportError:
        check("Kaggle Secrets", False, "not on Kaggle — skipping")
    all_ok &= secrets_ok

    # --- 4. MongoDB Atlas ---
    print("\n4. MongoDB Atlas connection")
    try:
        from kaggle_secrets import UserSecretsClient
        sc = UserSecretsClient()
        mongo_url = sc.get_secret("MONGODB_URL")
    except ImportError:
        mongo_url = os.environ.get("MONGODB_URL", "")

    if mongo_url:
        try:
            import asyncio
            from motor.motor_asyncio import AsyncIOMotorClient

            async def test_mongo():
                client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=10000)
                await client.admin.command("ping")
                # Try to create the openinsight database
                db = client["openinsight"]
                await db["test_collection"].insert_one({"test": True})
                await db["test_collection"].delete_many({})
                client.close()
                return True

            ok = asyncio.run(test_mongo())
            all_ok &= check("MongoDB reachable + writable", ok, "Atlas cluster connected")
        except Exception as e:
            all_ok &= check("MongoDB reachable", False, str(e)[:80])
    else:
        all_ok &= check("MongoDB URL", False, "not set")

    # --- 5. Zilliz Cloud ---
    print("\n5. Zilliz Cloud connection")
    try:
        from kaggle_secrets import UserSecretsClient
        sc = UserSecretsClient()
        zilliz_uri = sc.get_secret("ZILLIZ_URI")
        zilliz_token = sc.get_secret("ZILLIZ_TOKEN")
    except ImportError:
        zilliz_uri = os.environ.get("ZILLIZ_URI", os.environ.get("VECTOR_URI", ""))
        zilliz_token = os.environ.get("ZILLIZ_TOKEN", os.environ.get("VECTOR_TOKEN", ""))

    if zilliz_uri:
        try:
            from pymilvus import connections, utility
            connections.connect(alias="default", uri=zilliz_uri, token=zilliz_token)
            collections = utility.list_collections()
            all_ok &= check("Zilliz/Milvus reachable", True, f"connected ({len(collections)} existing collections)")
            connections.disconnect("default")
        except Exception as e:
            all_ok &= check("Zilliz/Milvus reachable", False, str(e)[:80])
    else:
        all_ok &= check("Zilliz URI", False, "not set")

    # --- 6. Disk space ---
    print("\n6. Disk space")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/kaggle/working" if is_kaggle else "/tmp")
        free_gb = free / 1e9
        all_ok &= check("Free disk space", free_gb > 5, f"{free_gb:.1f} GB free (need >5 GB for GROBID + repo + checkpoints)")
    except Exception as e:
        check("Disk space check", False, str(e)[:80])

    # --- 7. Internet (NCBI reachability) ---
    print("\n7. Internet connectivity")
    try:
        import httpx
        r = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pubmed", timeout=10)
        all_ok &= check("NCBI E-utilities", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        all_ok &= check("NCBI E-utilities", False, str(e)[:80])

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed — ready to run ingestion!")
        print("\nNext steps:")
        print("  1. Open kaggle_ingestion_v2.ipynb")
        print("  2. Edit the CONFIGURATION cell (SOURCES, LIMIT_PER_SOURCE)")
        print("  3. Run All")
        return 0
    else:
        print("❌ Some checks failed — fix the issues above before running ingestion.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
