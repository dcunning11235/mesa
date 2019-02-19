from abc import ABC
from mesa.agent import Agent
from typing import Any, Tuple, Set

Position = Any
Content = Any
GridCoordinate = Tuple[int, int]
CellContent = Set[Any]

class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __getitem__(self, pos: Position) -> Content:
        """Return the content of self at a position.
        Called by `_AbstractSpace()[pos]`.
        """

    @abstractmethod
    def __setitem__(self, pos: Position, content: Content) -> None:
        """Add content to self at position.
        Called by `_AbstractSpace()[pos] = content`.
        """

    @abstractmethod
    def __delitem__(self, content: Tuple[Position, Content]) -> None:
        """Delete content from the position in self.
        Called by `del _AbstractSpace()[pos, content]`.
        """

    def place_agent(self, agent: Agent, pos: Position) -> None:
        """Place an agent at a specific position."""

        self[pos] = agent
        setattr(agent, "pos", pos)

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""

        old_pos = getattr(agent, "pos")
        del self[old_pos, agent]
        setattr(agent, "pos", None)

    def move_agent(self, agent: Agent, pos: Position) -> None:
        """Move an agent from its current to a new position."""

        old_pos = getattr(agent, "pos")
        del self[old_pos, agent]
        self[pos] = agent
        setattr(agent, "pos", pos)

    @abstractmethod
    def content_at(self, pos: Position) -> Iterator[Content]:
        """Yield the content of a position."""

    @abstractmethod
    def agents_at(self, pos: Position) -> Iterator[Agent]:
        """Yield the agents at a specific position."""

    @abstractmethod
    def neighbors_of(self, agent: Agent) -> Iterator[Agent]:
        """Yield the neighbors of an Agent."""

    @abstractmethod
    def neighborhood_at(self, pos: Position, include_own: bool = True) -> Iterator[Position]:
        """Yield the neighborhood at a Position."""

    @abstractmethod
    def neighborhood_of(self, agent: Agent, include_own: bool = True) -> Iterator[Tuple[Position, Content]]:
        """Yield the neighborhood of an agent."""


class Grid(_AbstractSpace):
    def __init__(self, width: int, height: int):
        super().__init__()

        self.width = width
        self.height = height
        self._grid: Dict[GridCoordinate, CellContent] = dict()

    @property
    def default_value(self) -> CellContent:
        """Return the default value for empty cells."""
        return set()

    def __getitem__(self, pos: GridCoordinate) -> CellContent:
        try:
            return self._grid[pos]
        except KeyError:
            return self.default_value

    def __setitem__(self, pos: GridCoordinate, agent: Agent) -> None:
        try:
            self._grid[pos].add(agent)
        except KeyError:
            self._grid[pos] = {agent}

    def __delitem__(self, item: Tuple[GridCoordinate, Agent]) -> None:
        self._grid[item[0]].remove(item[1])
