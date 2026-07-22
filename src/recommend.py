"""Nearest-neighbor query + artist de-dup.

Not implemented yet.
"""


def recommend(query_vector, k=10, max_per_artist=1, exclude_track_id=None, exclude_artist=None):
    raise NotImplementedError


def recommend_from_file(path: str, k=10, max_per_artist=1):
    raise NotImplementedError


def recommend_from_search(query_str: str, k=10, max_per_artist=1):
    raise NotImplementedError
