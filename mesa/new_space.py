#from __future__ import annotations

from abc import ABC, abstractmethod
from mesa.agent import Agent
from typing import Any, Tuple, Set, Union, Optional, Callable, NamedTuple, Dict, \
    NewType, Iterator, Type, cast
from collections import namedtuple
from itertools import chain
from math import pow, sqrt
from inspect import stack
import sys

Position = Any
Content = Any
Distance = Union[int, float]

class GridCoordinate(NamedTuple):
    x: int
    y: int


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
    @classmethod
    @abstractmethod
    def distance(cls, pos1: Position, pos2: Position) -> Distance:
        """Returns the distance (real) bewteen two positions"""

    @classmethod
    @abstractmethod
    def path_length(cls, path: Iterator[Position]) -> Distance:
        """Returns the distance (real) along a given path/iterator of positions"""

    @classmethod
    @abstractmethod
    def neighborhood(cls, center: Position, radius: Distance) -> Union[Iterator[Position], '_AbstractSpace']:
        """Returns the neighborhood of a point with the given radius.  Returns an
        iterator of Position's if a discreet space, or a (sub)_AbstractSpace if
        continuous."""


class _NullMetric(_Metric):
    @classmethod
    def distance(cls, pos1: Position, pos2: Position) -> Distance:
        return 0

    @classmethod
    def path_length(cls, path: Iterator[Position]) -> Distance:
        return 0

    @classmethod
    def neighborhood(cls, center: Position, radius: Distance) -> Union[Iterator[Position], '_AbstractSpace']:
        return iter([])


class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[['_AbstractSpace', Position, Content], bool],
                metric: Union[_Metric, Type[_Metric]]) -> None:
        super().__init__()
        self.consistency_check = consistency_check
        self.metric = metric

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Return whether the space is continuous or discreet.  Discreet spaces
        are assumed to be finite, in addition."""

    @abstractmethod
    def __contains__(self, pos_or_content: Union[Position, Content]) -> bool:
        """Returns wether a *pos or content* is in the space."""

    @abstractmethod
    def get_all_positions(self) -> Iterator[Position]:
        """Returns an Iterator that gives all positions (for non-continuous
        spaces) or throws a TypeError."""

    @abstractmethod
    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Union[Iterator[Position], '_AbstractSpace']:
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
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool],
                metric: Union[_Metric, Type[_Metric]]) -> None:
        super().__init__(consistency_check, metric)
        self._agent_to_pos: Dict[Agent, Position] = {}

    def place_agent(self, pos: Position, agent: Agent) -> None:
        """Place an agent at a specific position."""
        # The order of ops here matters!
        self._agent_to_pos[agent] = pos
        self[pos] = agent

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        # The order of ops here matters!
        del self[agent]
        pos = self._agent_to_pos.pop(agent)

    def move_agent(self, pos: Position, agent: Agent) -> None:
        """Move an agent from its current to a new position."""
        self.remove_agent(agent)
        self.place_agent(pos, agent)

    def agent_exists(self, agent: Agent) -> bool:
        """Return if an agent exists within the space."""
        return agent in self._agent_to_pos

    def find_agent(self, agent: Agent) -> Optional[Position]:
        """Return where an agent is (its Position) within the space."""
        return self._agent_to_pos.get(agent, None)

    @property
    def agents(self) -> Iterator[Agent]:
        """Returns all agents within the space."""
        return self._agent_to_pos.keys()

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
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool],
                metric: Union[_Metric, Type[_Metric]],
                patch_name: str) -> None:
        super().__init__(consistency_check, metric)
        """Include path_name because for e.g. pure numeric patches there isn't
        any other identifying information."""
        self.patch_name = patch_name
        self.steps = 0

    @abstractmethod
    def __add__(self, other: Any) -> '_PatchSpace':
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __iadd__(self, other: Any) -> None:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __radd__(self, other: Any) -> None:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __sub__(self, other: Any) -> '_PatchSpace':
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
        super().__init__(None, _NullMetric)
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
    def __delitem__(self, pos_or_content: Union[LayeredPosition, Content]) -> None:
        if isinstance(pos_or_content, LayeredPosition):
            del self.layers[pos_or_content.layer][pos_or_content.pos]
        else:
            for l in self.layers.values():
                if isinstance(l, _AgentSpace) and pos_or_content in l:
                    del l[pos_or_content]

    def del_layer(self, layer_name: str) -> None:
        del self.layers[layer_name]

    # From _AgentSpace
    def __contains__(self, pos_or_content: Union[LayeredPosition, Content]) -> bool:
        if isinstance(pos_or_content, LayeredPosition):
            return pos_or_content.pos in self.layers[pos_or_content.layer]
        else:
            for l in self.layers.values():
                if isinstance(l, _AgentSpace):
                    if pos_or_content in l:
                        return True

        return False

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
            self._agent_to_layer[agent] = pos.layer
        else:
            raise TypeError("Cannot add agent to layer '{}' because it is not of type _AgentSpace".format(pos.layer))

    # From _AgentSpace
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        if isinstance(self.layers[self._agent_to_layer[agent]], _AgentSpace):
            cast(_AgentSpace, self.layers[self._agent_to_layer[agent]]).remove_agent(agent)
        else:
            raise TypeError("Something went wrong!  Cannot remove agent {} because \
                it is mapped to layer {}, which is not an _AgentSpace.  The layer \
                mapping has been removed.".format(agent, self._agent_to_layer[agent]))
        del self._agent_to_layer[agent]

    def agents_at(self, pos: Union[Position, LayeredPosition]) -> Iterator[Agent]:
        """Yield the agents at a specific position within a specific layer if
        the passed `pos` is of type `LayeredPosition`, else yield the agents at
        the passed `Position` for all layers"""
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _AgentSpace):
                return cast(_AgentSpace, self.layers[pos.layer]).agents_at(pos.pos)
            else:
                raise TypeError("Cannot return agents from layer '{}', it is not \
                    an instance of _AgentSpace.".format(pos.layer))
        else:
            return chain(*[l.agents_at(pos) for l in self.layers.values() if isinstance(l, _AgentSpace)])

    def neighbors_at(self, pos: Union[Position, LayeredPosition], radius: Distance = 1, include_own: bool = True) -> Iterator[Agent]:
        """Yield the agents in proximity to a position."""
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _AgentSpace):
                return cast(_AgentSpace, self.layers[pos.layer]).neighbors_at(pos.pos, radius, include_own)
            else:
                raise TypeError("Cannot get neighbors at pos '{}' in layer '{}' \
                    because it is not of type _AgentSpace".format(pos.pos, pos.layer))
        else:
            return chain(*[l.neighbors_at(pos, radius, include_own) for l in self.layers.values() if isinstance(l, _AgentSpace)])

    def neighbors_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Iterator[Agent]:
        """Yield the neighbors of an agent."""
        return cast(_AgentSpace, self.layers[self._agent_to_layer[agent]]).neighbors_of(agent, radius, include_own)

    def neighborhood_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Yield the neighborhood of an agent."""
        return cast(_AgentSpace, self.layers[self._agent_to_layer[agent]]).neighborhood_of(agent, radius, include_own)


class GridMetric(_Metric):
    @classmethod
    def path_length(cls, path: Iterator[Position]) -> Distance:
        ret: Distance = 0
        pos1: Position = next(path)
        for pos2 in path:
            ret += cls.distance(pos1, pos2)
            pos1 = pos2

        return ret


class EuclidianGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate) -> Distance:
        return sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance) -> Iterator[GridCoordinate]:
        # This is ugly and inefficient, but this will grind out the needed result
        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                if cls.distance((0, 0), (x, y)) <= radius:
                    yield (center[0]+x, center[1]+y)


class ManhattanGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate) -> Distance:
        return abs(coord1[0]-coord2[0]) + abs((coord1[1]-coord2[1]))

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance) -> Iterator[Position]:
        for y in range(-int(radius), int(radius)+1):
            for x in range(abs(y)-int(radius), int(radius)-abs(y)+1):
                yield (center[0]+x, center[1]+y)


class ChebyshevGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate) -> Distance:
        return max(abs(coord1[0]-coord2[0]), abs((coord1[1]-coord2[1])))

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance) -> Iterator[Position]:
        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                yield (center[0]+x, center[1]+y)


class GridConsistencyChecks:
    warn_flag = False

    @staticmethod
    def _get_caller() -> str:
        func_name = ""
        try:
            func_name = sys._getframe(2).f_code.co_name
        except:
            if not GridConsistencyChecks.warn_flag:
                GridConsistencyChecks.warn_flag = True
                raise ResourceWarning("sys._getframe(2).f_code.co_name is unavailable, using much slower inspect.stack()!")
            func_name = stack()[2].function

        if func_name == "":
            raise ValueError("Unable to get source method name for GridConsistencyChecks")

        return func_name

    @staticmethod
    def max1(grid: 'Grid', coord: GridCoordinate, agent: Agent) -> bool:
        caller = GridConsistencyChecks._get_caller()

        if caller == "__setitem__":
            return not len(grid[coord])

        return True

    @staticmethod
    def unique(grid: 'Grid', coord: GridCoordinate, agent: Agent) -> bool:
        caller = GridConsistencyChecks._get_caller()

        if caller == "__setitem__":
            return type(agent) not in map(type, grid[coord])

        return True


class Grid(_AgentSpace):
    def __init__(self,
                width: int, height: int, torus: bool,
                consistency_check: Callable[['Grid', GridCoordinate, Agent], bool] = GridConsistencyChecks.max1,
                metric: Union[_Metric, Type[_Metric]] = ChebyshevGridMetric):
        super().__init__(cast(Callable[[_AbstractSpace, Position, Content], bool], consistency_check), metric)
        self.width = width
        self.height = height
        self.torus = torus
        self._grid: Dict[GridCoordinate, CellContent] = dict()

    @property
    def default_value(self) -> Set:
        """Return the default value for empty cells."""
        return set()

    @property
    def is_continuous(self) -> bool:
        return False

    def _translate_coord(self, pos: GridCoordinate) -> GridCoordinate:
        return (pos[0] % self.width, pos[1] % self.height) if self.torus else pos

    def __getitem__(self, pos: GridCoordinate) -> set:
        try:
            return self._grid[self._translate_coord(pos)]
        except KeyError:
            return self.default_value

    def __setitem__(self, pos: GridCoordinate, agent: Agent) -> None:
        pos = self._translate_coord(pos)

        if self.consistency_check is not None:
            self.consistency_check(self, pos, agent)

        try:
            self._grid[pos].add(agent)
        except KeyError:
            self._grid[pos] = set([agent, ])

    def __delitem__(self, pos_or_content: Union[GridCoordinate, Agent]) -> None:
        if isinstance(pos_or_content, GridCoordinate):
            pos = self._translate_coord(pos_or_content)
            self._grid[pos].clear()
        else:
            self._grid[self._agent_to_pos[pos_or_content]].remove(pos_or_content)

    def __contains__(self, pos_or_content: Union[GridCoordinate, Agent]) -> bool:
        if isinstance(pos_or_content, GridCoordinate):
            pos = self._translate_coord(pos_or_content)
            return 0 <= pos[0] <= self.width and 0 <= pos[1] < self.height
        else:
            return pos_or_content in self._agent_to_pos

    def __missing__(self, pos: GridCoordinate) -> Set:
        if pos in self:
            return self.default_value

        LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[GridCoordinate]:
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            for n in cast(Iterator[Position], self.metric.neighborhood(pos, radius)):
                if 0 <= n[0] < self.width and 0 <= n[1] < self.height:
                    yield n
        else:
            raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        return iter(self[pos])

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Agent]:
        return chain(*[self.agents_at(pos) for n in self.neighborhood_at(pos, radius, include_own)])

    def get_all_positions(self) -> Iterator[Position]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)
