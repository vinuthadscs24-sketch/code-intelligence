from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """
    A node in a code-intelligence visualization.

    Examples:
        - method
        - class
        - file
        - database
        - notification
    """

    id: str
    label: Optional[str] = None
    type: str = "method"
    file: Optional[str] = None


class GraphEdge(BaseModel):
    """
    A directed relationship between two nodes.

    Examples:
        CALLS
        DEPENDS_ON
        WRITES_TO
        TRIGGERS
    """

    source: str
    target: str
    type: str = "CALLS"


class StructuredResponse(BaseModel):
    """
    Common response contract returned by the adaptive
    code-intelligence query layer.

    The frontend can use `response_type` to decide which
    visualization/panel should be rendered.
    """

    query: str

    response_type: str

    answer: str

    data: Dict[str, Any] = Field(default_factory=dict)

    evidence: List[Dict[str, Any]] = Field(default_factory=list)
