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
import collections
import numpy as np
import copy

Position = Any
Content = Any
Distance = Union[int, float]

'''
class GridCoordinate(NamedTuple):
    x: int
    y: int
'''
GridCoordinate = Tuple[int, int]

class LayeredPosition(NamedTuple):
    layer: str
    pos: Position


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
    def distance(cls, pos1: Position, pos2: Position, space: '_AbstractSpace') -> Distance:
        """Returns the distance (real) bewteen two positions"""

    @classmethod
    @abstractmethod
    def path_length(cls, path: Iterator[Position], space: '_AbstractSpace') -> Distance:
        """Returns the distance (real) along a given path/iterator of positions"""

    @classmethod
    @abstractmethod
    def neighborhood(cls, center: Position, radius: Distance, space: '_AbstractSpace') -> Union[Iterator[Position], '_AbstractSpace']:
        """Returns the neighborhood of a point with the given radius.  Returns an
        iterator of Position's if a discreet space, or a (sub)_AbstractSpace if
        continuous."""


class _NullMetric(_Metric):
    @classmethod
    def distance(cls, pos1: Position, pos2: Position, space: '_AbstractSpace') -> Distance:
        return 0

    @classmethod
    def path_length(cls, path: Iterator[Position], space: '_AbstractSpace') -> Distance:
        return 0

    @classmethod
    def neighborhood(cls, center: Position, radius: Distance, space: '_AbstractSpace') -> Union[Iterator[Position], '_AbstractSpace']:
        return iter([])


class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self,
                metric: Union[_Metric, Type[_Metric]],
                consistency_check: Callable[['_AbstractSpace', Position, Content], bool] = None) -> None:
        super().__init__()
        self.consistency_check = consistency_check
        self.metric = metric

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Return whether the space is continuous or discreet.  Discreet spaces
        are assumed to be finite, in addition."""

    @property
    @abstractmethod
    def default_value(self) -> Content:
        """Return whether the default value of the space.  For subclasses where
        __missing__ is called (or are infinite) this should be the value that
        intializes a point/cell/etc. such as set(), 0, None, etc."""

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
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Content]]:
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
                metric: Union[_Metric, Type[_Metric]],
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool]  = None) -> None:
        super().__init__(metric, consistency_check)
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
        return iter(self._agent_to_pos.keys())

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        """Yield the neighbors at a given position (i.e. neighnors_at, but with
        radius=0)."""
        return map(lambda tup: tup[1], self.neighbors_at(pos, 0, False))

    def neighbors_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Agent]]:
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
                patch_name: str,
                metric: Union[_Metric, Type[_Metric]],
                consistency_check: Callable[[_AbstractSpace, Position, Content], bool] = None) -> None:
        super().__init__(metric, consistency_check)
        """Include path_name because for e.g. pure numeric patches there isn't
        any other identifying information."""
        self.patch_name = patch_name
        self.steps = 0

    @abstractmethod
    def __add__(self, other: Any) -> '_PatchSpace':
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __iadd__(self, other: Any) -> '_PatchSpace':
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __radd__(self, other: Any) -> '_PatchSpace':
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    @abstractmethod
    def __sub__(self, other: Any) -> '_PatchSpace':
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __isub__(self, other: Any) -> '_PatchSpace':
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __rsub__(self, other: Any) -> '_PatchSpace':
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    @abstractmethod
    def __mul__(self, other: Any) -> '_PatchSpace':
        """Element-by-element multiply values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __imul__(self, other: Any) -> '_PatchSpace':
        """Element-by-element multiplication of values of one _PatchSpace by
        another _PatchSpace, scalar, etc."""

    @abstractmethod
    def __rmul__(self, other: Any) -> '_PatchSpace':
        """Element-by-element multiplication of values of one _PatchSpace by
        another _PatchSpace, scalar, etc."""

    @abstractmethod
    def __truediv__(self, other: Any) -> '_PatchSpace':
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __itruediv__(self, other: Any) -> '_PatchSpace':
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __pow__(self, other: Any) -> '_PatchSpace':
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def __ipow__(self, other: Any) -> '_PatchSpace':
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Content]]:
        """Yield the patches in proximity to a position, possible including those
        at the passed position."""

    @abstractmethod
    def step(self) -> None:
        """_PatchSpace is like model in that it has a step method, and like a
        BaseScheduler in that it has a self.steps that is incremented with each
        call of step.
        """
        self.steps += 1


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
        super().__init__(_NullMetric)
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

    def neighbors_at(self, pos: Union[Position, LayeredPosition], radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Agent]]:
        """Yield the agents in proximity to a position."""
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _AgentSpace):
                return cast(_AgentSpace, self.layers[pos.layer]).neighbors_at(pos.pos, radius, include_own)
            else:
                raise TypeError("Cannot get neighbors at pos '{}' in layer '{}' \
                    because it is not of type _AgentSpace".format(pos.pos, pos.layer))
        else:
            return chain(*[l.neighbors_at(pos, radius, include_own) for l in self.layers.values() if isinstance(l, _AgentSpace)])

    def neighbors_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Agent]]:
        """Yield the neighbors of an agent."""
        return cast(_AgentSpace, self.layers[self._agent_to_layer[agent]]).neighbors_of(agent, radius, include_own)

    def neighborhood_of(self, agent: Agent, radius: Distance = 1, include_own: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Yield the neighborhood of an agent."""
        return cast(_AgentSpace, self.layers[self._agent_to_layer[agent]]).neighborhood_of(agent, radius, include_own)


class GridMetric(_Metric):
    @classmethod
    @abstractmethod
    def distance(cls, pos1: GridCoordinate, pos2: GridCoordinate, space: _AbstractSpace) -> Distance:
        if pos1 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
        if pos2 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos2, space))

        return 0

    @classmethod
    def path_length(cls, path: Iterator[GridCoordinate], space: _AbstractSpace) -> Distance:
        ret: Distance = 0
        pos1: Position = next(path)
        if pos1 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
        for pos2 in path:
            if pos1 not in space:
                raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
            ret += cls.distance(pos1, pos2)
            pos1 = pos2

        return ret

    @classmethod
    @abstractmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace) -> Iterator[GridCoordinate]:
        if center not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(center, space))

        return iter([])


class EuclidianGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(EuclidianGridMetric, cls).distance(coord1, coord2, space)

        return sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace) -> Iterator[GridCoordinate]:
        super(EuclidianGridMetric, cls).neighborhood(center, radius, space)

        # This is ugly and inefficient, but this will grind out the needed result
        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                if cls.distance((0, 0), (x, y)) <= radius:
                    yield (center[0]+x, center[1]+y)


class ManhattanGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(ManhattanGridMetric, cls).distance(coord1, coord2, space)

        return abs(coord1[0]-coord2[0]) + abs((coord1[1]-coord2[1]))

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace) -> Iterator[Position]:
        super(ManhattanGridMetric, cls).neighborhood(center, radius, space)

        for y in range(-int(radius), int(radius)+1):
            for x in range(abs(y)-int(radius), int(radius)-abs(y)+1):
                yield (center[0]+x, center[1]+y)


class ChebyshevGridMetric(GridMetric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(ChebyshevGridMetric, cls).distance(coord1, coord2, space)

        return max(abs(coord1[0]-coord2[0]), abs((coord1[1]-coord2[1])))

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace) -> Iterator[Position]:
        super(ChebyshevGridMetric, cls).neighborhood(center, radius, space)

        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                yield (center[0]+x, center[1]+y)


class ConsistencyChecks:
    warn_flag = False

    @staticmethod
    def _get_caller() -> str:
        func_name = ""
        try:
            func_name = sys._getframe(2).f_code.co_name
        except:
            if not ConsistencyChecks.warn_flag:
                ConsistencyChecks.warn_flag = True
                raise ResourceWarning("sys._getframe(2).f_code.co_name is unavailable, using much slower inspect.stack()!")
            func_name = stack()[2].function

        if func_name == "":
            raise ValueError("Unable to get source method name for GridConsistencyChecks")

        return func_name

    @staticmethod
    def stack_checks(checks: Iterator[Callable[[_AbstractSpace, Position, Content], bool]]) -> Callable[[_AbstractSpace, Position, Content], bool]:
        def _check_stacker(space: _AbstractSpace, pos: Position, content: Content):
            for check in checks:
                if check(space, pos, content):
                    continue
                return False

        return _check_stacker



class AgentConsistencyChecks(ConsistencyChecks):
    @staticmethod
    def max1(space: _AgentSpace, coord: GridCoordinate, agent: Agent) -> bool:
        caller = ConsistencyChecks._get_caller()

        if caller == "__setitem__":
            val = space[coord]
            if isinstance(val, collections.Iterator):
                return not len(list(val))

        return True

    @staticmethod
    def unique(space: _AgentSpace, coord: GridCoordinate, agent: Agent) -> bool:
        caller = ConsistencyChecks._get_caller()

        if caller == "__setitem__":
            val = space[coord]
            if isinstance(val, collections.Iterator):
                return type(agent) not in map(type, space[coord])

        return True

# Need to create a e.g. Cartesian abstract class that is a space indexed by (x, y)???
# class _Grid()...

class Grid(_AgentSpace):
    def __init__(self,
                width: int, height: int, torus: bool,
                metric: Union[_Metric, Type[_Metric]] = ChebyshevGridMetric,
                consistency_check: Callable[[_AgentSpace, GridCoordinate, Agent], bool] = AgentConsistencyChecks.max1):
        super().__init__(metric, cast(Callable[[_AbstractSpace, Position, Content], bool], consistency_check))
        self.width = width
        self.height = height
        self.torus = torus
        self._grid: Dict[GridCoordinate, Set] = dict()

    @property
    def default_value(self) -> Set:
        return set()

    @property
    def is_continuous(self) -> bool:
        return False

    def _verify_coord(self, pos: GridCoordinate, raise_exception: bool = True) -> Optional[GridCoordinate]:
        if 0 <= pos[0] <= self.width and 0 <= pos[1] < self.height:
            return pos
        elif self.torus:
            return (pos[0] % self.width, pos[1] % self.height)
        elif raise_exception:
            raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

        return None

    def __getitem__(self, pos: GridCoordinate) -> set:
        return self._grid.get(cast(GridCoordinate, self._verify_coord(pos)), self.default_value)

    def __setitem__(self, pos: GridCoordinate, agent: Agent) -> None:
        pos = cast(GridCoordinate, self._verify_coord(pos))

        if self.consistency_check is not None and self.consistency_check(self, pos, agent):
            try:
                self._grid[pos].add(agent)
            except KeyError:
                self._grid[pos] = set([agent, ])
        else:
            raise ValueError("Cannot add agent {} to position {}, failed consistency check {}".format(agent, pos, self.consistency_check))

    def __delitem__(self, pos_or_content: Union[GridCoordinate, Agent]) -> None:
        if isinstance(pos_or_content, tuple):
            pos = cast(GridCoordinate, self._verify_coord(pos))
            self._grid[pos].clear()
        else:
            self._grid[self._agent_to_pos[pos_or_content]].remove(pos_or_content)

    def __contains__(self, pos_or_content: Union[GridCoordinate, Agent]) -> bool:
        if isinstance(pos_or_content, tuple):
            return self._verify_coord(pos_or_content, False) is not None
        else:
            return pos_or_content in self._agent_to_pos

    def __missing__(self, pos: GridCoordinate) -> Set:
        if pos in self:
            return self.default_value

        raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

    def neighborhood_at(self, pos: GridCoordinate, radius: Distance = 1, include_own: bool = True) -> Iterator[GridCoordinate]:
        pos = cast(GridCoordinate, self._verify_coord(pos))
        for n in cast(Iterator[Position], self.metric.neighborhood(pos, radius, self)):
            yield n

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        return iter(self[pos])

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[GridCoordinate, Agent]]:
        return chain(*[map(lambda a: (pos, a), self.agents_at(pos)) for n in self.neighborhood_at(pos, radius, include_own)])

    def get_all_positions(self) -> Iterator[Position]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)


class PatchConsistencyChecks(ConsistencyChecks):
    @staticmethod
    def gte0(space: _PatchSpace, coord: GridCoordinate, value: Content) -> bool:
        caller = ConsistencyChecks._get_caller()

        if caller == "__setitem__":
            return value >=0

        return True

# An altrnative path would be to have NumpyPatchGrid actually extend numpy.ndarray
# but I have some doubts. E.g. implementing consistency checks would require wrapping
# most methods, less so but still with torus, etc.  That is all "fine", though it
# doesn't save work.  Worse, would have to disable e.g. being able to reshape,
# in-place sort, and changing of various lower-level flags etc.  But it would be
# neat if we just extended it...
class NumpyPatchGrid(_PatchSpace):
    def __init__(self,
            patch_name: str, init_val: np.ndarray, torus: bool,
            metric: Union[_Metric, Type[_Metric]] = ChebyshevGridMetric,
            consistency_check: Callable[[_PatchSpace, GridCoordinate, Agent], bool] = PatchConsistencyChecks.gte0):
        super().__init__(patch_name, metric, cast(Callable[[_AbstractSpace, Position, Content], bool], consistency_check))
        self.torus = torus
        if init_val.ndim != 2:
            raise TypeError("NumericPatchGrid may only be initilialized with a ndarray of dimension 2")
        self._grid = np.array(init_val)  # We don't need no stinkin' matrices here
        self.height, self.width = self._grid.shape

    def _verify_other(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> Union[np.ndarray, int, float]:
        if isinstance(other, NumpyPatchGrid) or isinstance(other, np.ndarray):
            if isinstance(other, NumpyPatchGrid):
                ret = other._grid
            else:
                ret = other

            if self._grid.shape != ret.shape:
                raise TypeError("Incompatiable shape for passed grid, must be {}".format(self._grid.shape))
        else:
            ret = other

        return ret

    def __add__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        ret = copy.copy(self)
        ret += other
        return ret

    def __iadd__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        self._grid += self._verify_other(other)
        return self

    __radd__ = __add__

    def __sub__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        ret = copy.copy(self)
        ret -= other
        return ret

    def __isub__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        self._grid -= self._verify_other(other)
        return self

    __rsub__ = __sub__

    def __mul__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        ret = copy.copy(self)
        ret *= other
        return ret

    def __imul__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        self._grid *= self._verify_other(other)
        return self

    __rmul__ = __mul__

    def __truediv__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        ret = copy.copy(self)
        ret /= other
        return ret

    def __itruediv__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        self._grid /= self._verify_other(other)
        return self

    __rtruediv__ = __truediv__

    def __pow__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        ret = copy.copy(self)
        ret **= other
        return ret

    def __ipow__(self, other: Union['NumpyPatchGrid', np.ndarray, int, float]) -> 'NumpyPatchGrid':
        self._grid **= self._verify_other(other)
        return self

    @property
    def is_continuous(self) -> bool:
        return False

    # ####Exactly the same as Grid!!!!
    def _verify_coord(self, pos: GridCoordinate, raise_exception: bool = True) -> Optional[GridCoordinate]:
        if 0 <= pos[0] <= self.width and 0 <= pos[1] < self.height:
            return pos
        elif self.torus:
            return (pos[0] % self.width, pos[1] % self.height)
        elif raise_exception:
            raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

        return None

    def __contains__(self, pos: GridCoordinate) -> bool:
        return self._verify_coord(pos, False) is not None

    # ####Exactly the same as Grid!!!!
    def get_all_positions(self) -> Iterator[GridCoordinate]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def __getitem__(self, pos: GridCoordinate) -> Content:
        return self._grid[self._verify_coord(pos)]

    def __setitem__(self, pos: GridCoordinate, content: Content) -> None:
        pos = cast(GridCoordinate, self._verify_coord(pos))

        if self.consistency_check is not None and self.consistency_check(self, pos, content):
            self._grid[pos] = content
        else:
            raise ValueError("Cannot set value {} to position {}, failed consistency check {}".format(content, pos, self.consistency_check))

    def __delitem__(self, pos: GridCoordinate) -> None:
        pos = cast(GridCoordinate, self._verify_coord(pos))

        if self.consistency_check is not None and self.consistency_check(self, pos, self.default_value):
            self._grid[pos] = self.default_value
        else:
            raise ValueError("Cannot delete value {} at position {}, failed consistency check {}".format(self._grid[pos], pos, self.consistency_check))

    # ####Exactly the same as Grid!!!!
    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[GridCoordinate]:
        if 0 <= pos[0] < self.width and 0 <= pos[1] < self.height:
            for n in cast(Iterator[Position], self.metric.neighborhood(pos, radius)):
                if 0 <= n[0] < self.width and 0 <= n[1] < self.height:
                    yield n
        else:
            raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_own: bool = True) -> Iterator[Tuple[Position, Content]]:
        return map(lambda pos: (pos, self._grid[pos]), self.neighborhood_at(pos, radius, include_own))
