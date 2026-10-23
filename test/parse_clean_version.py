from packaging.version import Version, InvalidVersion
import re

def parse_clean_version(version_str: str) -> Version:
    # Normalize "-development" or any variation like "-dev" to ".dev0"
    cleaned = re.sub(r'[-_.]?development', '.dev0', version_str, flags=re.IGNORECASE)
    # Catch cases where it's just raw text hyphens (e.g., 1.0.0-dev -> 1.0.0.dev0)
    cleaned = re.sub(r'-dev(\d*)', r'.dev\1', cleaned, flags=re.IGNORECASE)
    try:
        return Version(cleaned)
    except InvalidVersion:
        # Fallback regex to pull just the numerical digits if it's still breaking
        match = re.search(r'(\d+(\.\d+)+)', version_str)
        if match:
            return Version(match.group(1))
        raise
