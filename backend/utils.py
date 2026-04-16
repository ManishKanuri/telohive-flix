def build_item_text(row) -> str:
    title    = str(row.get("title",        "") or "").strip()
    ctype    = str(row.get("type",         "") or "").strip()
    genres   = str(row.get("listed_in",    "") or "").strip()
    desc     = str(row.get("description",  "") or "").strip()
    director = str(row.get("director",     "") or "").strip()
    cast     = str(row.get("cast",         "") or "").strip()
    country  = str(row.get("country",      "") or "").strip()
    year     = str(row.get("release_year", "") or "").strip()
    rating   = str(row.get("rating",       "") or "").strip()

    parts = []

    if title and ctype:
        parts.append(f"{title} is a {ctype}")
    elif title:
        parts.append(title)

    if genres and year:
        parts.append(f"in the genres of {genres}, released in {year}")
    elif genres:
        parts.append(f"in the genres of {genres}")
    elif year:
        parts.append(f"released in {year}")

    if rating:
        parts.append(f"rated {rating}")

    if desc:
        parts.append(desc)

    if director:
        parts.append(f"Directed by {director}")
    if cast:
        top = ", ".join(c.strip() for c in cast.split(",")[:5] if c.strip())
        if top:
            parts.append(f"Starring {top}")

    if country:
        parts.append(f"from {country}")

    return ". ".join(parts)
