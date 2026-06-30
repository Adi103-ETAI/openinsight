"""Verify reachability of all source websites.

Run this from your LOCAL machine (not the sandbox) to check if the
government sites are actually reachable. The sandbox where Phase 0-3
was built has network egress restrictions that block many .gov.in and
.nic.in domains — those sites may work fine from your machine.

Usage:
    python scripts/verify_sites.py

Exit code 0 = all sites reachable
Exit code 1 = some sites unreachable (see output for details)
"""
from __future__ import annotations

import sys
import socket
from urllib.parse import urlparse

import httpx


# All sites we've built scrapers for, with alternative URLs to try
SITES: list[tuple[str, str, list[str]]] = [
    # (source_name, primary_url, [alternative_urls])
    ("pubmed", "https://eutils.ncbi.nlm.nih.gov", []),
    ("statpearls", "https://www.ncbi.nlm.nih.gov/books/", []),
    ("ncbi_bookshelf", "https://www.ncbi.nlm.nih.gov/books/", []),
    ("medknow", "https://www.medknow.com", []),
    
    # Indian gov sites — these are the ones to verify
    ("nmc_curriculum", "https://www.nmc.org.in", []),
    ("cdsco", "https://cdsco.gov.in", [
        "https://cdscoonline.gov.in",
        "http://cdsco.nic.in",
    ]),
    ("ctri", "http://ctri.nic.in", [
        "https://ctri.nic.in",
    ]),
    ("pvpi", "https://www.ipc.gov.in", [
        "http://ipc.gov.in",
    ]),
    ("ntep", "https://tbcindia.gov.in", [
        "http://tbcindia.gov.in",
        "https://ntep.nic.in",
        "http://tbcindia.nic.in",
    ]),
    ("nvbdcp", "https://nvbdcp.gov.in", [
        "http://nvbdcp.gov.in",
        "https://nvbdcp.nic.in",
    ]),
    ("nhm", "https://nhm.gov.in", [
        "http://nrhm.gov.in",
    ]),
    ("npcds", "https://npcdc.nic.in", [
        "http://npcpcds.nic.in",
        "https://npcdc.gov.in",
    ]),
    ("indmed", "https://indmedinfo.nic.in", [
        "http://indmedinfo.nic.in",
        "https://medind.nic.in",
        "http://medind.nic.in",
    ]),
    
    # Phase 4 — specialty societies
    ("rssdi", "https://rssdi.in", []),
    ("csi", "https://csi-india.org", []),
    ("iap", "https://www.iapindia.org", []),
    ("fogsi", "https://www.fogsi.org", []),
    ("isccm", "https://isccm.org", []),
]


def check_site(name: str, primary_url: str, alt_urls: list[str]) -> dict:
    """Check one site + its alternative URLs. Returns result dict."""
    result = {
        "name": name,
        "primary_url": primary_url,
        "primary_ok": False,
        "working_url": None,
        "alternatives_tried": [],
        "error": None,
    }
    
    urls_to_try = [primary_url] + alt_urls
    for url in urls_to_try:
        try:
            with httpx.Client(timeout=15, follow_redirects=True, verify=False, headers={
                "User-Agent": "OpenInsight-Bot/0.1 (connectivity check; contact: hello@openinsight.in)"
            }) as c:
                r = c.get(url)
                if r.status_code < 500:
                    if url == primary_url:
                        result["primary_ok"] = True
                    result["working_url"] = url
                    return result
                else:
                    result["alternatives_tried"].append(f"{url} → HTTP {r.status_code}")
        except httpx.ConnectTimeout:
            result["alternatives_tried"].append(f"{url} → timeout")
        except httpx.ConnectError as e:
            result["alternatives_tried"].append(f"{url} → {str(e)[:50]}")
        except Exception as e:
            result["alternatives_tried"].append(f"{url} → {type(e).__name__}")
    
    result["error"] = "All URLs failed"
    return result


def main() -> int:
    print("=" * 80)
    print("OpenInsight — Source Website Connectivity Check")
    print("=" * 80)
    print()
    print(f"Checking {len(SITES)} sources...")
    print()
    
    all_ok = True
    results = []
    
    for name, primary_url, alt_urls in SITES:
        result = check_site(name, primary_url, alt_urls)
        results.append(result)
        
        if result["primary_ok"]:
            status = "OK"
            detail = ""
        elif result["working_url"]:
            status = "ALT"
            detail = f"→ use {result['working_url']}"
            all_ok = False  # primary failed, need to update config
        else:
            status = "FAIL"
            detail = f"→ tried: {'; '.join(result['alternatives_tried'][:2])}"
            all_ok = False
        
        print(f"  {name:<20} {status:<5} {primary_url:<40} {detail}")
    
    print()
    print("=" * 80)
    
    failures = [r for r in results if not r["primary_ok"]]
    if not failures:
        print("✅ All sites reachable. Scrapers should work.")
        return 0
    
    alts = [r for r in failures if r["working_url"]]
    dead = [r for r in failures if not r["working_url"]]
    
    if alts:
        print(f"\n⚠️  {len(alts)} sites need URL updates (primary failed, alternative worked):")
        for r in alts:
            print(f"   {r['name']}: change {r['primary_url']} → {r['working_url']}")
    
    if dead:
        print(f"\n❌ {len(dead)} sites are unreachable (all URLs failed):")
        for r in dead:
            print(f"   {r['name']}:")
            for attempt in r["alternatives_tried"]:
                print(f"     {attempt}")
        print()
        print("For dead sites, you need to:")
        print("  1. Visit the site in a browser to confirm it's really down")
        print("  2. Search for the current URL (gov sites often move)")
        print("  3. Update the SourceConfig in src/ingestion/scrapers/sources/<name>.py")
    
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
