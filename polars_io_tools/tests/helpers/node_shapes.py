"""``NodeShape``: what a parsed filter node (from ``io_sources.base.get_parsed_expr``) should look like, for tests that
check the tree a filter parses into."""

from polars_io_tools.io_sources.base import BaseExprNode


class NodeShape:
    """What a parsed node should look like: its class and the attributes to compare.

    An attribute value that is a ``NodeShape`` or a list of them is matched the same way, a function is called with the
    attribute and must return ``True``, ``None`` and booleans are compared by identity, and anything else by equality.
    """

    def __init__(self, cls: type[BaseExprNode], **attrs):
        self.cls, self.attrs = cls, attrs

    def match(self, node, path: str = "node") -> None:
        assert isinstance(node, self.cls), f"{path}: expected {self.cls.__name__}, got {type(node).__name__}"
        for name, want in self.attrs.items():
            match_shape(getattr(node, name), want, f"{path}.{name}")


def match_shape(got, want, path: str) -> None:
    """Assert ``got`` matches ``want``: a ``NodeShape``, a list of them, a function returning ``True``, or a value."""
    if isinstance(want, NodeShape):
        want.match(got, path)
    elif isinstance(want, list) and want and isinstance(want[0], NodeShape):
        assert len(got) == len(want), f"{path}: {len(got)} items, expected {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            match_shape(g, w, f"{path}[{i}]")
    elif callable(want) and not isinstance(want, type):
        assert want(got), f"{path}: {got!r} fails {want}"
    elif want is None or isinstance(want, bool):
        assert got is want, f"{path}: {got!r} is not {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"
