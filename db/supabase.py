import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger("db.supabase")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    if not SUPABASE_URL or not SUPABASE_URL.startswith("http"):
        raise ValueError(f"Invalid or missing SUPABASE_URL: {SUPABASE_URL!r}")
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Missing SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
except Exception as e:
    logger.error("CRITICAL: Failed to initialize Supabase client: %s. Ensure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set in environment variables.", e)
    try:
        supabase = create_client("https://placeholder.supabase.co", "placeholder-key")
    except Exception:
        supabase = None