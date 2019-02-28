from abc import ABC
from mesa.agent import Agent
from typing import Any, Tuple, Set, Union, Optional, Callable, NamedTuple
from number import Number, Real
from collections import namedtuple
from itertools import chain
from math import pow, sqrt
from inspect import stack

Position = Any
Content = Any

GridCoordinate = Tuple[int, int]
CellContent = Set[Any]

class _Metric(ABC):
    @staticmethod
    @abstractmethod
    def distance(pos1: Position, pos2: Position) -> Real:
        """Returns the distance (real) bewteen two positions"""

    @staticmethod
    @abstractmethod
    def neighborhood(center: Position, radius: Real) -> Union[Iterator[Position], _AbstractSpace]:
        """Returns the neighborhood of a point with the given radius.  Returns an
        iterator of Position's if a discreet space, or a (sub)_AbstractSpace if
        continuous."""


class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self, consistency_check: Callable[[_AbstractSpace, Position, Content], bool] = None,
                metric: _Metric) -> None:
        super().__init__()
        self.consisitency_check = consisitency_check
        self.metric = metric

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Return whether the space is continuous or discreet."""

    '''
    Getting way to complicated here, leave this until much later...

    @property
    @abstractmethod
    def is_cell_fixed(self) -> Optional[bool]:
        """Return whether cells of a discreet space are fixed in size, None if
        the space is continuous."""

    @property
    @abstractmethod
    def cell_dimensions(self) -> Union[Number, Tuple[Number, ...], None]:
        """Return the cell dimensions if the cell size is fixed and the space is
        discreet."""
    '''

    @abstractmethod
    def neighborhood_at(self, pos: Position, radius: Number = 1, include_own: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Yield the neighborhood at a position, either an iterator over the
        positions or an _AbstractSpace containing only the subspace of the
        neighborhood."""

    @abstractmethod
    def __getitem__(self, pos: Position) -> Content:
        """Return the content or value of self at a position.
        Called by `_AbstractSpace()[pos]`."""

    @abstractmethod
    def __setitem__(self, pos: Position, content: Content) -> None:
        """Set the content or value at a position.
        Called by `_AbstractSpace()[pos] = content`."""


class _AgentSpace(_AbstractSpace):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_AgentSpace, Position, Agent], bool] = None,
                metric: _Metric) -> None:
        super().__init__(consisitency_check, metric)

    @abstractmethod
    def __delitem__(self, content: Tuple[Position, Agent]) -> None:
        """Delete content from the position in self.
        Called by `del _AbstractSpace()[pos, content]`."""

    @abstractmethod
    def __contains__(self, content: Tuple[Position, Agent]) -> bool:
        """Determine if Agent is contained at Position in this space."""

    def place_agent(self, agent: Agent, pos: Position) -> None:
        """Place an agent at a specific position."""

        self[pos] = agent
        setattr(agent, "pos", pos)

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""

        old_pos = getattr(agent, "pos")
        if (old_pos, agent) not in self:
            raise KeyError("Agent not removed because {} was not found at position {}.".format(agent, old_pos))
        del self[old_pos, agent]
        setattr(agent, "pos", None)

    def move_agent(self, agent: Agent, pos: Position) -> None:
        """Move an agent from its current to a new position."""

        self.remove_agent(agent)
        self.place_agent(agent, pos)

    @abstractmethod
    def agents_at(self, pos: Position) -> Iterator[Agent]:
        """Yield the agents at a specific position."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Number = 1) -> Iterator[Agent]:
        """Yield the agents in proximity to a position."""

    @abstractmethod
    def neighbors_of(self, agent: Agent, radius: Number = 1) -> Iterator[Agent]:
        """Yield the neighbors of an agent."""

    @abstractmethod
    def neighborhood_of(self, agent: Agent, radius: Number = 1, include_own: bool = True) -> Union[Iterator[Position], AgentSpace]:
        """Yield the neighborhood of an agent."""


class _PatchSpace(_AbstractSpace):
    @abstractmethod
    def __init__(self, consistency_check: Callable[[_PatchSpace, Position, Content], bool] = None,
                metric: _Metric,
                patch_name: str = None, patch_type: type = None) -> None:
        super().__init__(consisitency_check, distance, neighborhood)
        self.patch_name = patch_name
        self.patch_type = patch_type
        self.steps = 0

    @abstractmethod
    def __setitem__(self, pos: Position, content: Content) -> None:
        """Set the content or value at a position.
        Called by `_AbstractSpace()[pos] = content`."""
        if self.patch_type is not None and isinstance(content, self.patch_type):
            raise AttributeError("Cannot assign type of {} to Patch of type {}.".format(type(content), self.patch_type))

    @abstractmethod
    def __add__(self, other: Any) -> _PatchSpace:
        """Add values of one _PatchSpace to another _PatchSpace"""

    @abstractmethod
    def __iadd__(self, other: Any) -> None:
        """Add values of one _PatchSpace to another _PatchSpace"""

    @abstractmethod
    def __sub__(self, other: Any) -> _PatchSpace:
        """Subtract values of one _PatchSpace from another _PatchSpace"""

    @abstractmethod
    def __isub__(self, other: Any) -> None:
        """Subtract values of one _PatchSpace from another _PatchSpace"""

    @abstractmethod
    def step(self) -> None:
        """_PatchSpace is like model in that it has a step method, and like a
        BaseScheduler in that it has a self.steps that is incremented with each
        call of step.
        """
        self.steps += 1


class LayeredPosition(NamedTuple):
    layer: str
    pos: Position

# Relies on __getitem__  on _AbstractSpace implementations not throwing
# KeyError's but returning defaults, BUT ALSO doing their own bounds
# checking and throwing errors appropriately.
class LayeredSpace(_AgentSpace, _PatchSpace):
    def __init__(self):
        super().__init__()
        self.layers: Dict[str, _AbstractSpace] = {}

    # From _AbstractSpace
    def __getitem__(self, pos: LayeredPosition) -> Content:
        return self.layers[pos.layer][pos.pos]

    # From _AbstractSpace
    def __setitem__(self, pos: LayeredPosition, content: Content) -> None:
        self.layers[pos.layer][pos.pos] = content

    # From _AgentSpace
    def __delitem__(self, content: Tuple[LayeredPosition, Content]) -> None:
        self.layers[content[0].layer].__delitem__( (content[0].pos, content[1]) )

    # From _AgentSpace
    @abstractmethod
    def __contains__(self, content: Tuple[LayeredPosition, Agent]) -> bool:
        return (content[0].pos, content[1]) in self.layers[content[0].layer]

    # From _AgentSpace
    def place_agent(self, agent: Agent, pos: LayeredPosition) -> None:
        """Place an agent at a specific position."""
        self.layers[pos.layer.place_agent(agent, pos.pos)
        setattr(agent, "layer", pos[0])

    # From _AgentSpace
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        self.layers[getattr(agent, "layer")].remove_agent(agent, getattr(agent, "pos"))
        setattr(agent, "layer", None)

    def agents_at(self, pos: Union[Position, LayeredPosition]) -> Iterator[Agent]:
        """Yield the agents at a specific position within a specific layer if
        the passed `pos` is of type `LayeredPosition`, else yield the agents at
        the passed `Position` for all layers"""
        if instanceof(pos, LayeredPosition):
            return self.layers[pos.layer].agents_at(pos.pos)
        else:
            return chain(*[l.agents_at(pos) for l in self.layers.values if instanceof(l, _AgentSpace)])

    def neighbors_at(self, pos: Union[Position, LayeredPosition], radius: Number = 1) -> Iterator[Agent]:
        """Yield the agents in proximity to a position."""
        if instanceof(pos, LayeredPosition):
            return self.layers[pos.layer].neighbors_at(pos.pos, radius)
        else:
            return chain(*[l.neighbors_at(pos, radius) for l in self.layers.values if instanceof(l, _AgentSpace)])

    def neighbors_of(self, agent: Agent, radius: Number = 1) -> Iterator[Agent]:
        """Yield the neighbors of an agent."""
        return self.layers[getattr(agent, "layer")].neighbors_of(agent, radius)

    def neighborhood_of(self, agent: Agent, radius: Number = 1, include_own: bool = True) -> Union[Iterator[LayeredPosition], AgentSpace]:
        """Yield the neighborhood of an agent."""
        return self.layers[getattr(agent, "layer")].neighborhood_of(agent, radius, include_own)


class EuclidianGridMetric(_Metric):
    @staticmethod
    def distance(coord1: GridCoordinate, coord2: GridCoordinate) -> Real:
        return sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

    @staticmethod
    def neighborhood(center: GridCoordinate, radius: int) -> Iterator[GridCoordinate]:
        # This is ugly and inefficient, but this will grind out the needed result
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                if sqrt(y**2 + x**2) <= radius:
                    yield (center[0]+x, center[1]+y)


class ManhattanGridMetric(_Metric):
    @staticmethod
    def manhattan(coord1: GridCoordinate, coord2: GridCoordinate) -> Real:
        return abs(coord1[0]-coord2[0]) + abs((coord1[1]-coord2[1]))

    @staticmethod
    def neighborhood(center: GridCoordinate, radius: int) -> Iterator[Position]:
        for y in range(-radius, radius+1):
            for x in range(abs(y)-radius, radius-abs(y)+1):
                yield (center[0]+x, center[1]+y)


class ChebyshevGridMetric(_Metric):
    @staticmethod
    def chebyshev(coord1: GridCoordinate, coord2: GridCoordinate) -> Real:
        return max(abs(coord1[0]-coord2[0]), abs((coord1[1]-coord2[1])))

    @staticmethod
    def neighborhood(center: GridCoordinate, radius: int) -> Iterator[Position]:
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                yield (center[0]+x, center[1]+y)


class GridConsistencyChecks:
    warn_flag = False

    @staticmethod
    def _get_caller() -> str:
        func_name = ""
        try:
            func_name = sys._getframe(2).f_code.co_name
        except:
            if not warn_flag:
                warn_flag = True
                raise ResourceWarning("sys._getframe(2).f_code.co_name is unavailable, using much slower inspect.stack()!")
            func_name = inspect.stack()[2].function

        if func_name == "":
            raise ValueException("Unable to get source method name for GridConsistencyChecks")

        return func_name

    @staticmethod
    def max1(grid: Grid, coord: GridCoordinate, agent: Agent) -> bool:
        caller = _get_caller()

        if caller == "__setitem__":
            return not len(grid[coord])

    @staticmethod
    def unique(grid: Grid, coord: GridCoordinate, agent: Agent) -> bool:
        caller = _get_caller()

        if caller == "__setitem__":
            return type(agent) not in map(type, grid[coord])


class Grid(_AgentSpace):
    def __init__(self, width: int, height: int,
                consistency_check: Callable[[Grid, GridCoordinate, Agent, str], bool] = ConsistencyChecks.max1,
                metric: _Metric = ChebyshevGridMetric:
        super().__init__(consisitency_check, metric)

        self.width = width
        self.height = height
        self._grid: Dict[GridCoordinate, CellContent] = dict()

    @property
    def default_value(self) -> CellContent:
        """Return the default value for empty cells."""
        return set()

    @property
    def is_continuous(self) -> bool:
        return False

    def __getitem__(self, pos: GridCoordinate) -> set:
        try:
            return self._grid[pos]
        except KeyError:
            return self.default_value

    def __setitem__(self, pos: GridCoordinate, agent: Agent) -> None:
        if self.consistency_check is not None:
            self.consistency_check(self, pos, agent)

        try:
            self._grid[pos].add(agent)
        except KeyError:
            self._grid[pos] = {agent}

    def __delitem__(self, item: Tuple[GridCoordinate, Agent]) -> None:
        self._grid[item[0]].remove(item[1])

    def __contains__(self, item: Tuple[GridCoordinate, Agent]) -> bool:
        return item[1] in self._grid[item[0]]

    def neighborhood_at(self, pos: GridCoordinate, radius: int = 1, include_own: bool = True) -> Iterator[GridCoordinate]:
        """Yield the neighborhood at a position, either an iterator over the
        positions or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            for n in self.metric.neighborhood(pos, radius):
                if 0 <= n[0] < self.width and 0 <= n[1] < self.height:
                    yield n

    def agents_at(self, pos: GridCoordinate) -> Iterator[Agent]:
        return iter(self._grid[pos])

    def neighbors_at(self, pos: GridCoordinate, radius: Number = 1) -> Iterator[Agent]:
        neighborhood_at

    def neighbors_of(self, agent: Agent, radius: Number = 1) -> Iterator[Agent]:
        """Yield the neighbors of an agent."""

    def neighborhood_of(self, agent: Agent, radius: Number = 1, include_own: bool = True) -> Union[Iterator[GridCoordinate], AgentSpace]:
        """Yield the neighborhood of an agent."""
