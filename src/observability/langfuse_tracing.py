import os
from contextlib import contextmanager


DEFAULT_TAGS = ["sklearn-rag-agent", "v2_structured"]
_missing_config_reported = False


def _get_langfuse_host():
    return os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")


def is_langfuse_configured():
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and _get_langfuse_host()
    )


def _report_missing_config_once():
    global _missing_config_reported

    if not _missing_config_reported:
        print(
            "Langfuse tracing is not configured; continuing without traces. "
            "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST to enable it."
        )
        _missing_config_reported = True


def get_langfuse_client():
    if not is_langfuse_configured():
        _report_missing_config_once()
        return None

    host = _get_langfuse_host()
    if host and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = host

    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:
        print(f"Langfuse tracing could not be initialized; continuing without traces: {exc}")
        return None


def flush_langfuse(client):
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            print(f"Langfuse flush failed; continuing without blocking execution: {exc}")


def chunk_metadata(points):
    metadata = []

    for point in points:
        payload = point.payload or {}
        metadata.append(
            {
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "topic": payload.get("topic"),
                "section": payload.get("section"),
                "url": payload.get("url"),
                "score": point.score,
            }
        )

    return metadata


@contextmanager
def trace_context(client, name, input_data=None, metadata=None, tags=None):
    if client is None:
        yield None
        return

    combined_tags = list(dict.fromkeys(DEFAULT_TAGS + (tags or [])))

    try:
        manager = client.start_as_current_span(
            name=name,
            input=input_data,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"Langfuse trace creation failed; continuing without traces: {exc}")
        yield None
        return

    with manager as span:
        try:
            client.update_current_trace(
                name=name,
                input=input_data,
                metadata=metadata,
                tags=combined_tags,
            )
        except Exception as exc:
            print(f"Langfuse trace update failed; continuing without blocking execution: {exc}")

        yield span


@contextmanager
def observation(client, name, input_data=None, metadata=None, as_type="span"):
    if client is None:
        yield None
        return

    try:
        span = client.start_observation(
            name=name,
            as_type=as_type,
            input=input_data,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"Langfuse observation creation failed; continuing without traces: {exc}")
        yield None
        return

    try:
        yield span
    finally:
        try:
            span.end()
        except Exception as exc:
            print(f"Langfuse observation end failed; continuing without blocking execution: {exc}")


def update_observation(span, output=None, metadata=None):
    if span is not None:
        try:
            span.update(output=output, metadata=metadata)
        except Exception as exc:
            print(f"Langfuse observation update failed; continuing without blocking execution: {exc}")
