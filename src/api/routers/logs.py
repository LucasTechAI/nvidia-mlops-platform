"""
Logs Router — structured log access with SQLite backend.

Provides endpoints for querying, filtering, and aggregating
application logs stored in the database.  Also keeps legacy
file-based endpoints for raw log access.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/logs", tags=["logs"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _db():
    from src.utils.log_database import LogDatabase
    return LogDatabase.get_instance()


# ════════════════════════════════════════════════════════════════════
#  Structured (DB) endpoints — used by the new charts frontend
# ════════════════════════════════════════════════════════════════════

@router.get("/stats")
async def get_log_stats(
    since: int = Query(120, description="Minutes to look back", ge=1, le=10080),
):
    """Aggregate statistics: totals, by level, by source, rates."""
    stats = _db().get_stats(since_minutes=since)
    return {
        "total": stats.total,
        "by_level": stats.by_level,
        "by_source": stats.by_source,
        "error_rate": stats.error_rate,
        "warning_rate": stats.warning_rate,
        "logs_per_minute": stats.logs_per_minute,
        "since_minutes": since,
    }


@router.get("/timeline")
async def get_log_timeline(
    since: int = Query(120, description="Minutes to look back", ge=1, le=10080),
):
    """Time-bucketed log counts for line/area charts."""
    timeline = _db().get_timeline(since_minutes=since)
    return {"timeline": timeline, "since_minutes": since}


@router.get("/entries")
async def get_log_entries(
    level: Optional[str] = Query(None, description="Filter by level"),
    source: Optional[str] = Query(None, description="Filter by source"),
    search: Optional[str] = Query(None, description="Search in message"),
    since: int = Query(120, description="Minutes to look back", ge=1, le=10080),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Paginated log entries with optional filters."""
    entries = _db().query(
        level=level,
        source=source,
        search=search,
        since_minutes=since,
        limit=limit,
        offset=offset,
    )
    return {
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "level": e.level,
                "source": e.source,
                "module": e.module,
                "message": e.message,
                "extra": e.extra,
            }
            for e in entries
        ],
        "count": len(entries),
        "limit": limit,
        "offset": offset,
    }


@router.get("/sources")
async def get_log_sources():
    """List all distinct log sources in the database."""
    return {"sources": _db().get_sources()}


@router.post("/cleanup")
async def cleanup_old_logs(keep_hours: int = Query(72, ge=1)):
    """Remove old log entries."""
    deleted = _db().cleanup(keep_hours=keep_hours)
    return {"deleted": deleted, "keep_hours": keep_hours}


# ════════════════════════════════════════════════════════════════════
#  Legacy file-based endpoints (kept for backwards compatibility)
# ════════════════════════════════════════════════════════════════════

@router.get("/api")
async def get_api_logs():
    """Get API server logs from file."""
    try:
        api_log_path = Path("/tmp/api.log")
        if api_log_path.exists():
            with open(api_log_path, "r") as f:
                lines = f.readlines()
                content = "".join(lines[-1000:])
        else:
            content = "API log file not available."
        return {"content": content, "lines": len(content.split("\n"))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training")
async def get_training_logs():
    """Get model training logs from file."""
    try:
        logs_dir = PROJECT_ROOT / "logs"
        log_files = []
        if logs_dir.exists():
            log_files = sorted(logs_dir.glob("training_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not log_files:
                log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            with open(log_files[0], "r") as f:
                lines = f.readlines()
                content = "".join(lines[-1000:])
        else:
            content = "No training logs found."
        return {"content": content, "lines": len(content.split("\n")), "file": str(log_files[0]) if log_files else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def get_services_logs():
    """Get logs from Docker services."""
    try:
        services = []
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for container in [n for n in result.stdout.strip().split("\n") if n]:
                    try:
                        lr = subprocess.run(
                            ["docker", "logs", "--tail", "500", container],
                            capture_output=True, text=True, timeout=5,
                        )
                        services.append({"name": container, "logs": lr.stdout + lr.stderr})
                    except Exception as e:
                        services.append({"name": container, "logs": f"Error: {e}"})
        except FileNotFoundError:
            pass
        if not services:
            services.append({"name": "info", "logs": "No running services found."})
        return {"services": services}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_logs():
    """Get general system logs from files."""
    try:
        parts = []
        logs_dir = PROJECT_ROOT / "logs"
        if logs_dir.exists():
            for lf in sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                with open(lf, "r") as f:
                    lines = f.readlines()
                    parts.append(f"=== {lf.name} ===\n" + "".join(lines[-500:]))
        return {"content": "\n\n".join(parts) or "No system logs available.", "lines": sum(len(p.split("\n")) for p in parts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
