def normalize_internal_url(url: str | None) -> str | None:
    if not url:
        return None

    stripped_url = url.strip()
    if (
        not stripped_url.startswith("/")
        or stripped_url.startswith("//")
        or "\\" in stripped_url
        or any(char.isspace() for char in stripped_url)
    ):
        return None
    return stripped_url
