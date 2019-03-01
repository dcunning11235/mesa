from abc import ABC, abstractmethod
from mesa.agent import Agent
from typing import Any, Tuple, Set, Union, Optional, Callable, NamedTuple, Dict, \
    NewType, Iterator, SupportsInt, SupportsFloat, cast
from collections import namedtuple
from itertools import chain
from math import pow, sqrt
from inspect import stack

Position = Any
Content = Any
Distance = Union[SupportsInt, SupportsFloat]

GridCoordinate = Tuple[int, int]
CellContent = Set[Any]


class _Metric(ABC):
    """Class that implements path length, distance (two-point path), and
    neighborhood for _AbstractSpace.  The idea here is roughly similar to
    distance and a 'ball' in e.g. topology.  However, for irregular/path-dependent
    spaces, `distance` cannot be gauranteed to return the true distance (aka, the
    shortest path) between two points (much less amongst more than 2), and hence
    `neighborhood` also cannot be gauranteed to return the true ball.

    For simple measures, e.g. a 2-D grid or continuous space with a Chebyshev or
    Euclidian metric, this should of course of exact.
    """
    @staticmethod
    @abstractmethod
    def distance(pos1: Position, pos2: Position) -> Distance:
        """Returns the distance (real) bewteen two positions"""

    @staticmethod
    @abstractmethod
    def path_length(path: Iterator[Position]) -> Distance:
        """Returns the distance (real) along a given path/iterator of positions"""

    @staticmethod
    @abstractmethod
    def neighborhood(center: Position, radius: Distance) -> Union[Iterator[Position], _AbstractSpace]:
        """Returns the neighborhood of a point with the given radius.  Returns an
        iterator of Position's if a discreet space, or a (sub)_AbstractSpace if
        continuous."""


class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool] = None,
                metric: _Metric = None) -> None:
        super().__init__()
        self.consistency_check = consistency_check
        self.metric = metric

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Return whether the space is continuous or discreet.  Discreet spaces
        are assumed to be finite, in addition."""

    @abstractmethod
    def __contains__(self, pos: Position) -> bool:
        """Returns wether a pos is in the space."""

    @abstractmethod
    def get_all_positions(self) -> Iterator[Position]:
        """Returns an Iterator that gives all positions (for non-continuous
        spaces) or throws a TypeError."""

    @abstractmethod
    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Yield the neighborhood at a position, either an iterator over the
        positions or an _AbstractSpace containing only the subspace of the
        neighborhood."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Content]:
        """Yield the neighbors in proximity to a position, possible including those
        at the passed position."""

    @abstractmethod
    def __getitem__(self, pos: Position) -> Union[Iterator[Content], Content]:
        """Return the content or value at a position, *or an iterator over such.*
        Called by `_AbstractSpace()[pos]`."""

    @abstractmethod
    def __setitem__(self, pos: Position, content: Content) -> None:
        """*Sets or adds* the content or value at a position.
        Called by `_AbstractSpace()[pos] = content`."""

    @abstractmethod
    def __delitem__(self, pos_or_content: Union[Position, Content]) -> None:
        """Delete content or value at a position.  This should *not* remove the
        position itself (e.g. unlike `del dict[key]`).  E.g. a Grid implementation
        should should still return some 'empty' value for coordinates that are
        in-bounds.  See `__missing__`."""

    @abstractmethod
    def __missing__(self, pos: Position) -> Content:
        """Handle missing positions.  Used for e.g. lazy filling.  Should raise
        an appropriate exception for e.g. out-of-bounds positions."""


class _AgentSpace(_AbstractSpace):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool] = None,
                metric: _Metric = None) -> None:
        super().__init__(consistency_check, metric)
        self._agent_to_pos: Dict[Agent, Position] = {}

    def place_agent(self, pos: Position, agent: Agent) -> None:
        """Place an agent at a specific position."""
        self[pos] = agent
        self._agent_to_pos[agent] = pos

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        pos = self._agent_to_pos.pop(agent)
        del self[agent]

    def move_agent(self, pos: Position, agent: Agent) -> None:
        """Move an agent from its current to a new position."""
        self.remove_agent(agent)
        self.place_agent(pos, agent)

    @abstractmethod
    def agent_exists(self, agent: Agent) -> bool:
        """Return if an agent exists within the space."""

    @abstractmethod
    def find_agent(self, agent: Agent) -> Position:
        """Return where an agent is (its Position) within the space."""

    @property
    @abstractmethod
    def agents(self) -> Iterator[Agent]:
        """Returns all agents within the space."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Agent]:
        """Yield the agents in proximity to a position, possible including those
        at the passed position."""

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        """Yield the neighbors at a given position (i.e. neighnors_at, but with
        radius=0)."""
        return self.neighbors_at(pos, 0, False)

    def neighbors_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Iterator[Agent]:
        """Yield the agents that are the neighbors of a given agent, including
        itself possibly."""
        return self.neighbors_at(self.find_agent(agent), radius, include_own)

    def neighborhood_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Yield the positions of the neighborhood of an agent."""
        return self.neighborhood_at(self.find_agent(agent), radius, include_own)


class _PatchSpace(_AbstractSpace):
    """_PatchSpace holds simple values, or wraps objects that present a simple
    value, which can be +,-,*,/, or ** together or with a scalar.  A `step`
    method is also included so that the _PatchSpace iself can be e.g. added to a
    scheduler.
    """
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool] = None,
                metric: _Metric = None,
                patch_name: str = None) -> None:
        super().__init__(consistency_check, metric)
        """Include path_name because for e.g. pure numeric patches there isn't
        any other identifying information."""
        self.patch_name = patch_name
        self.steps = 0

    @abstractmethod
    def __add__(self, other: Any) -> _PatchSpace:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __iadd__(self, other: Any) -> None:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __radd__(self, other: Any) -> None:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __sub__(self, other: Any) -> _PatchSpace:
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __isub__(self, other: Any) -> None:
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __rsub__(self, other: Any) -> None:
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __mul__(self, other: Any) -> None:
        """Element-by-element multiply values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __imul__(self, other: Any) -> None:
        """Element-by-element multiplication of values of one _PatchSpace by
        another _PatchSpace, scalar, etc."""

    @abstractmethod
    def __rmul__(self, other: Any) -> None:
        """Element-by-element multiplication of values of one _PatchSpace by
        another _PatchSpace, scalar, etc."""

    @abstractmethod
    def __div__(self, other: Any) -> None:
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __idiv__(self, other: Any) -> None:
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __pow__(self, other: Any) -> None:
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __ipow__(self, other: Any) -> None:
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Content]]:
        """Yield the agents in proximity to a position, possible including those
        at the passed position."""

    @abstractmethod
    def value_at(self, pos: Position) -> Content:
        """Yield the value at a given position."""

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
class LayeredSpace(_AgentSpace):
    """
    LayeredSpace is a composite of _AbstractSpace's, each named.  Adding an
    _AgentSpace to a LayeredSpace causes its __setitem__, __delitem__,
    place_agent, and remove_agent methods to be wrapped.
    """
    def __init__(self, layers: Dict[str, _AbstractSpace] = {}):
        super().__init__()
        self.layers: Dict[str, _AbstractSpace] = layers
        self._agent_to_layer: Dict[Agent, str] = {}

    # From _AbstractSpace
    def __getitem__(self, pos: LayeredPosition) -> Content:
        return self.layers[pos.layer][pos.pos]

    def get_layer(self, layer: str) -> _AbstractSpace:
        return self.layers[layer]

    # From _AbstractSpace
    def __setitem__(self, pos: LayeredPosition, content: Content) -> None:
        self.layers[pos.layer][pos.pos] = content

        del self._agent_to_layer[content]
        if isinstance(content, Agent):
            self._agent_to_layer[content] = pos.layer

    def set_layer(self, layer_name:str, layer: _AbstractSpace) -> None:
        old_layer = self.layers[layer_name] if layer_name in self.layers else None
        self.layers[layer_name] = layer

        if old_layer is not None and isinstance(old_layer, _AgentSpace):
            for agent in old_layer.agents:
                del self._agent_to_layer[agent]
        if isinstance(layer, _AgentSpace):
            for agent in layer.agents:
                self._agent_to_layer[agent] = layer_name


    # From _AgentSpace
    def __delitem__(self, content: LayeredPosition) -> None:
        del self.layers[content.layer][content.pos]

    def del_layer(self, layer_name: str) -> None:
        del self.layers[layer_name]

    # From _AgentSpace
    def __contains__(self, pos: LayeredPosition) -> bool:
        return pos.pos in self.layers[pos.layer]

    def contains_layer(self, layer_name: str) -> bool:
        return layer_name in self.layers

    def get_common_positions(self) -> Iterator[Position]:
        """This could potentially be a very expensive call.  It depends entirely
        on what the layers are doing, how they ar eimplemented, how large they
        are, etc."""
        continuous_spaces: Set = set()
        common_discreet_positions: Set = set()
        found_discreet: bool = False

        for layer in self.layers.values():
            if not layer.is_continuous:
                if not found_discreet:
                    common_discreet_positions |= set(layer.get_all_positions())
                elif len(common_discreet_positions) == 0:
                    break
                else:
                    common_discreet_positions &= set(layer.get_all_positions())
            else:
                continuous_spaces.add(layer)

        if found_discreet and len(common_discreet_positions) == 0:
            return
        elif found_discreet:
            for pos in common_discreet_positions:
                if len(continuous_spaces) > 0:
                    miss = False
                    for space in continuous_spaces:
                        if pos not in space:
                            miss = True
                            break
                    if not miss:
                        yield pos
                else:
                    yield pos
        else:
            raise NotImplementedError("Left medium (to heavy..?) lifting until later")

    # From _AgentSpace
    def place_agent(self, pos: LayeredPosition, agent: Agent) -> None:
        """Place an agent at a specific position."""
        if isinstance(self.layers[pos.layer], _AgentSpace):
            cast(_AgentSpace, self.layers[pos.layer]).place_agent(agent, pos.pos)
        else:
            raise TypeError("Cannot add agent to layer '{}' because it is not of type _AgentSpace".format(pos.layer))


############################################Pickup#################################



    # From _AgentSpace
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        self.layers[getattr(agent, "layer")].remove_agent(agent, getattr(agent, "pos"))

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
                metric: _Metric = ChebyshevGridMetric):
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
