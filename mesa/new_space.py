from __future__ import annotations

from abc import ABC, abstractmethod
from mesa.agent import Agent
from typing import Any, Tuple, Set, List, Union, Optional, Callable, NamedTuple, \
    Dict, Iterator, Type, cast
from collections import namedtuple
from itertools import chain
from math import pow, sqrt
from inspect import stack
import sys
import collections
import numpy as np
import copy
import networkx as nx
from networkx.generators.ego import ego_graph

Position = Any
Content = Any

Distance = Union[int, float]

GridCoordinate = Tuple[int, int]
ContinuousCoordinate = Tuple[float, float]

class LayeredPosition(NamedTuple):
    layer: str
    pos: Union[Position, Content]


# Might use something like this to create a factory for e.g. _Metrics
# This is not used anywhere yet, this is just...  a scratchpad copy.  A note.
class UsableByMeta(type):
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        else:
            usable_by = getattr(cls, '__usable_by', None)
            if target__for is not None:
                if not isinstance(target__for, collections.Iterable):
                    cls.registry[target__for] = (cls)
                else:
                    for tf in target__for:
                        cls.registry[tf] = (cls)


# Relying on Metric to do two actually different things... to allow _Metric
# subclasses to get e.g. neighborhood, they may need to hold information, i.e.
# be instantiated.  Example:  A network metric that needs to know the structure
# (and possibly e.g. edge weights) to compute distances and neighborhoods.  This
# can be done by passing the _AbstractSpace or by instantiating.  For something
# like my first pass at NetworkX, what makes more sense?
#
# The question of instantiating versus passing everytime to a static/classmethod
# hinges, for me, on what the relation between the _AbstractSpace, the _Metric,
# and the underlying implementation.  E.g. for the NetworkX space, the _Metric
# needs the actual networkx Graph to compute neighbors and distances.  So the
# Space calls the distance method on the Metric which calls the distance method
# on the Grpah... held by the Space.  Seems less than optimal.
#
# That said, the Graph object of the Space also holds the e.g. Agents.  So putting
# it into the Metric seems misplaced, to say the least.
#
# What seems needed is a _Metric which knows about the underlying structure,
# an _Space wrapper for that strucutre, and a storage/retrieval layer that can
# put things on/into the space, query the metric,
class _Metric(ABC):
    """Class that implements path length, distance (two-point path), path, and
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
    def distance(cls, pos1: Position, pos2: Position, space: _AbstractSpace) -> Distance:
        """Returns the distance (int or real) bewteen two positions or raises an
        exception if no such distance exists"""
        if pos1 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
        if pos2 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos2, space))

        return 0

    @classmethod
    def path_length(cls, path: Iterator[Position], space: _AbstractSpace) -> Distance:
        """Returns the distance (real) along a given path/iterator of positions,
        or raises an exception if not such length exists (one or more points
        don't exist, a path is not possible/could nbot be found, etc.)"""
        ret: Distance = 0
        pos1: Position = next(path)
        if pos1 not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
        for pos2 in path:
            if pos1 not in space:
                raise LookupError("path_length failed because '{}' is not in space {}".format(pos1, space))
            ret += cls.distance(pos1, pos2, space)
            pos1 = pos2

        return ret

    @classmethod
    @abstractmethod
    def path(cls, pos1: Position, pos2: Position, space: _AbstractSpace) -> Iterator[Position]:
        """Returns a path bewteen two positions, one that (perhaps not uniquely)
        corresponds with the distance returned by `distance`.  For a continuous
        space this will be an approximation of a path by some number of points,
        minimally `(pos1, pos2)`.  If no path is possible (or can be found),
        an exception will be raised."""

    @classmethod
    @abstractmethod
    def neighborhood(cls, center: Position, radius: Distance, space: _AbstractSpace, include_center: bool = True) -> Union[Iterator[Position], _AbstractSpace]:
        """Returns the neighborhood of a point with the given radius.  Returns an
        iterator of Position's if a discreet space, or a (sub)_AbstractSpace if
        continuous, or None if no neighborhood can be found (e.g. the `center`
        is invalid.)"""
        if center not in space:
            raise LookupError("path_length failed because '{}' is not in space {}".format(center, space))

        return iter([])


# I think I really need to step back here and think about the idea of a network vs
# a space.  A network of agents is actually a very different thing than a network
# or grid or "space" of positons that Agents may (or may not) occupy, singularly
# or in multiple.  Treating Agents as (potential) Positions just about solves this,
# but I think there there may in fact be plenty here to have to trunks of a class
# hierarchy, an _AgentNetwork and an _PositionalAgentSpace trunk, each of which share a
# fairly minimal base class (__contain__, neighborhood_at, and possibly all of
# __get/set/delitem__).  I need to think about this more tomorrow.
class _AbstractSpace(ABC):
    @abstractmethod
    def __init__(self, metric: Type[_Metric]) -> None:
        super().__init__()
        self.metric = metric

    @abstractmethod
    def __contains__(self, pos_or_content: Union[Position, Content]) -> bool:
        """Returns whether a position or content is in the space."""

    @abstractmethod
    def __iter__(self) -> Iterator[Union[Position, Content]]:
        """Returns an iterator over all keys (positions or content)"""

    @abstractmethod
    def neighbors(self, pos_or_content: Union[Position, Content], radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Content]:
        """Yield the neighbors in proximity to a position or content, possible including those
        at the passed position."""


class _PositionalSpace(_AbstractSpace):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_PositionalSpace, Position, Content], bool] = None,
                **kwargs) -> None:
        super().__init__(**kwargs)
        self.consistency_check = consistency_check

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Return whether the space is continuous or discreet."""

    @property
    @abstractmethod
    def default_value(self) -> Optional[Content]:
        """Return whether the default value of the space.  For subclasses where
        __missing__ is called (or are infinite) this should be the value that
        intializes a point/cell/etc. such as set(), 0, None, etc."""

    @abstractmethod
    def content(self) -> Iterator[Content]:
        """Returns an Iterator that gives all content, flattened if e.g. multiple
        items are stored at each location."""

    @abstractmethod
    def all(self) -> Iterator[Tuple[Position, Content]]:
        """Returns an Iterator that gives all (Position, Content) tuples, flattened
        if e.g. multiple items are stored at each location."""

    @abstractmethod
    def __getitem__(self, pos: Position) -> Union[Iterator[Content], Content]:
        """Return the content at a position, or an iterator over such.
        Called by `_AbstractSpace()[pos]`."""

    @abstractmethod
    def __setitem__(self, pos: Position, content: Content) -> None:
        """*Sets or adds* the content at a position.
        Called by `_AbstractSpace()[pos] = content`."""

    @abstractmethod
    def __delitem__(self, pos_or_content: Union[Position, Content]) -> None:
        """Delete content or value at a position.  This should *not* remove the
        position itself (e.g. unlike `del dict[key]`).  E.g. a Grid implementation
        should should still return some 'empty' value for coordinates that are
        in-bounds.  See `__missing__`."""

    @abstractmethod
    def __missing__(self, pos: Position) -> Optional[Content]:
        """Handle missing positions.  Used for e.g. lazy filling.  Should raise
        an appropriate exception for e.g. out-of-bounds positions."""

    @abstractmethod
    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_pos_or_content: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood at a position, either an iterator over the
        keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""

    @abstractmethod
    def neighborhood_of(self, content: Content, radius: Distance = 1, include_pos_or_content: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood around some content, either an iterator
        over the keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Content]:
        """Yield the neighbors at/around a position as an iterator over the
        neighbors."""

    @abstractmethod
    def neighbors_of(self, content: Content, radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Content]:
        """Yield the neighbors around some content as an iterator over the
        neighbors."""


class _SocialSpace(_AbstractSpace):
    @abstractmethod
    def __init__(self,
                consistency_check: Callable[[_SocialSpace, Content], bool] = None,
                **kwargs) -> None:
        super().__init__(**kwargs)
        self.consistency_check = consistency_check


class _PositionalAgentSpace(_PositionalSpace):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_to_pos: Dict[Agent, Position] = {}

    def place_agent(self, pos: Position, agent: Agent) -> None:
        """Place an agent at a specific position."""
        self[pos] = agent

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        del self[agent]

    def move_agent(self, pos: Position, agent: Agent) -> None:
        """Move an agent from its current to a new position."""
        self.remove_agent(agent)
        self.place_agent(pos, agent)

    def find_agent(self, agent: Agent) -> Optional[Position]:
        """Return where an agent is (its Position) within the space."""
        return self._agent_to_pos.get(agent, None)

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        """Yield the neighbors at a given position (i.e. neighnors_at, but with
        radius=0)."""
        return self.neighbors_at(pos, 0, True)

    @abstractmethod
    def count_agents_at(self, pos: Position) -> int:
        """Yield the neighbors at a given position (i.e. neighnors_at, but with
        radius=0)."""

    @abstractmethod
    def content(self) -> Iterator[Content]:
        """Returns an Iterator that gives all content, flattened if e.g. multiple
        items are stored at each location."""
        return iter(self._agent_to_pos.keys())

    @abstractmethod
    def all(self) -> Iterator[Tuple[Position, Content]]:
        """Returns an Iterator that gives all (Position, Content) tuples, flattened
        if e.g. multiple items are stored at each location."""
        return map(lambda tup: (tup[1], tup[0]), self._agent_to_pos.items())

    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood at a position or content, either an iterator
        over the keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        return self.neighbors(pos, radius, include_center)

    def neighborhood_of(self, content: Content, radius: Distance = 1, include_self: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood at a position or content, either an iterator
        over the keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        # Technically there is a bug here... if continuous and multi-agent, no way
        # currently to return that... Just resturns the ball, including central
        # point, hence including `self`
        if self.is_continuous or include_self:
            return self.neighbors(self.find_agent(content), radius, True)

        self_pos = self.find_agent(content)
        for a in self.neighbors(self.find_agent(content), radius, True):
            if self_pos != a:
                yield a

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Iterator[Content]:
        """Yield the neighborhood at a position or content, either an iterator
        over the keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        return chain(*[self.agents_at(n) for n in self.neighborhood_at(pos, radius, include_center)])

    def neighbors_of(self, content: Content, radius: Distance = 1, include_self: bool = True) -> Iterator[Content]:
        """Yield the neighborhood at a position or content, either an iterator
        over the keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        return self.neighbors(content, radius, include_self)


class _SocialAgentSpace(_SocialSpace):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class _PatchSpace(_AbstractSpace):
    """_PatchSpace holds simple values, or wraps objects that present a simple
    value, which can be +,-,*,/,%,&,|,^ or ** together or with a scalar.  A `step`
    method is also included so that the _PatchSpace iself can be e.g. added to a
    scheduler.

    _PatchSpaces do not exist 'on their own'.  They are instantiated based on
    another space, from which they take their metric and other properties
    (dimensions, indexing, etc.)

    Subclasses must e.g. implement how they are initialized, what backs them, etc.
    """
    @abstractmethod
    def __init__(self, base_space: _AbstractSpace) -> None:
        super().__init__(base_space.metric)
        self.base_space = base_space
        self.steps = 0

    def __add__(self, other: Any) -> _PatchSpace:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""
        ret = copy.copy(self)
        ret += other
        return ret

    @abstractmethod
    def __iadd__(self, other: Any) -> _PatchSpace:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    __radd__ = __add__

    def __sub__(self, other: Any) -> _PatchSpace:
        """Add values of one _PatchSpace, scalar, etc. to another _PatchSpace"""
        ret = copy.copy(self)
        ret -= other
        return ret

    @abstractmethod
    def __isub__(self, other: Any) -> _PatchSpace:
        """Subtract values of one _PatchSpace, scalar, etc. from another _PatchSpace"""

    __rsub__ = __sub__

    def __mul__(self, other: Any) -> _PatchSpace:
        """Element-by-element multiply values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""
        ret = copy.copy(self)
        ret *= other
        return ret

    @abstractmethod
    def __imul__(self, other: Any) -> _PatchSpace:
        """Element-by-element multiplication of values of one _PatchSpace by
        another _PatchSpace, scalar, etc."""

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _PatchSpace:
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""
        ret = copy.copy(self)
        ret /= other
        return ret

    @abstractmethod
    def __itruediv__(self, other: Any) -> _PatchSpace:
        """Element-by-element division of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    def __mod__(self, other: Any) -> _PatchSpace:
        """Element-by-element mod of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""
        ret = copy.copy(self)
        ret %= other
        return ret

    @abstractmethod
    def __imod__(self, other: Any) -> _PatchSpace:
        """Element-by-element mode of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    def __pow__(self, other: Any) -> _PatchSpace:
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""
        ret = copy.copy(self)
        ret **= other
        return ret

    @abstractmethod
    def __ipow__(self, other: Any) -> _PatchSpace:
        """Element-by-element power of values of one _PatchSpace by another
        _PatchSpace, scalar, etc."""

    def __and__(self, other: Any) -> _PatchSpace:
        """And values of one _PatchSpace, scalar, etc. to another _PatchSpace"""
        ret = copy.copy(self)
        ret &= other
        return ret

    @abstractmethod
    def __iand__(self, other: Any) -> _PatchSpace:
        """And values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    __rand__ = __and__

    def __or__(self, other: Any) -> _PatchSpace:
        """Or values of one _PatchSpace, scalar, etc. to another _PatchSpace"""
        ret = copy.copy(self)
        ret |= other
        return ret

    @abstractmethod
    def __ior__(self, other: Any) -> _PatchSpace:
        """Or values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    __ror__ = __or__

    def __xor__(self, other: Any) -> _PatchSpace:
        """Xor values of one _PatchSpace, scalar, etc. to another _PatchSpace"""
        ret = copy.copy(self)
        ret ^= other
        return ret

    @abstractmethod
    def __ixor__(self, other: Any) -> _PatchSpace:
        """Xor values of one _PatchSpace, scalar, etc. to another _PatchSpace"""

    __rxor__ = __xor__

    @abstractmethod
    def step(self) -> None:
        """_PatchSpace is like model in that it has a step method, and like a
        BaseScheduler in that it has a self.steps that is incremented with each
        call of step.
        """
        self.steps += 1

    def __contains__(self, pos_or_content: Union[Position, Content]) -> bool:
        """Returns whether a position or content is in the space."""
        return pos_or_content in self.base_space

    def __iter__(self) -> Iterator[Union[Position, Content]]:
        """Returns an iterator over all keys (positions or content)"""
        return iter(self.base_space)

    @abstractmethod
    def neighbors(self, pos_or_content: Union[Position, Content], radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Union[Position, Content]]:
        """Yield the neighbors in proximity to a position or content, in this case
        the patch values, possibly including those at the passed position."""
        return self.base_space.neighbors(pos_or_content, radius, include_pos_or_content)


class _PositionalPatchSpace(_PositionalSpace, _PatchSpace):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def is_continuous(self) -> bool:
        return cast(_PositionalSpace, self.base_space).is_continuous

    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood at a position, either an iterator over the
        keys or an _AbstractSpace containing only the subspace of the
        neighborhood."""
        return cast(_PositionalSpace, self.base_space).neighborhood_at(pos, radius, include_center)

    def neighborhood_of(self, content: Content, radius: Distance = 1, include_self: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood around some content *in the underlying
        self.base_space*, either an iterator over the keys or an _AbstractSpace
        containing only the subspace of the neighborhood."""
        return cast(_PositionalSpace, self.base_space).neighborhood_of(content, radius, include_self)

    @abstractmethod
    def neighbors_at(self, pos: Position, radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Tuple[Position, Content]]:
        """Yield the patch neighbors at/around a position as an iterator over the
        neighbors.  Note that since patches may (likely) be simple types e.g.
        int's, the iterator is over (Positon, Content) tuples. """

    @abstractmethod
    def neighbors_of(self, content: Content, radius: Distance = 1, include_pos_or_content: bool = True) -> Iterator[Tuple[Position, Content]]:
        """Yield the patch neighbors around some content *in the underlying
        self.base_space* as an iterator over the neighbors.  Note that since
        patches may (likely) be simple types e.g. int's, the iterator is over
        (Positon, Content) tuples."""


# How to layer spaces that rely on different topologies.  E.g. a Grid full of
# farmers, a grid of merchants, a couple of corresponging GridPatchSpace's, and...
# a SocialNetwork of cities.  We can force the network type.  That is easy
# solution, so that it isn't a SocialNetwork at all but a PositionalNetwork.
# Otherwise, need some kind of mapping, where CityAgent "Wheatopia" maps to (7,1)
# and CityAgent "Cornucopia" maps to (35,122)... But at that point... doesn't it
# just make sense to change the type of the CityAgents' network?  I suppose it
# could be a case of, "Yeah, but it was already done this way, we don't want to."
#
# Have to think more on this.
#
# Possible solutions:
# (1) Force the re-implementation to PositionalNetwork <--------- DOING THIS
# (2) Include some kind of lookup/mapper as part of or utility class passed to
#     LayeredSpace
# (3) Include a wrapper, e.g. a "NetworkToPositional" wrapper that "converts" the
#     class (and would include a tranlsator... but not as part of LayeredSpace.)
#
# This also involves how LayeredSpaces are created.  Are they created (or
# updated) by appending spaces on, like:
#     stack = LayeredSpace(layers = {some layers}, order = [some order])  <--- YES
#     stack.set_layer('foo1', foo1_layer, z=-7) <--- NO
#     stack.set_layer('foo2', foo2_layer, z=2) <--- NO
#     stack.del_layer('bar4') <--- NO
# This seems unessesary.  After all, the model structure, the layers, is set
# at construction time...
#
# In that case, it should be like:
#      stack = LayeredSpace(layers = {some layers}, order = [some order])
# or maybe better yet:
#      stack = LayeredSpace.new(layers = {some layers}, order = [some order])
# where new could check for layer types and produce an appropriate LayeredSpace
# subclass instance, as needed.
#
class LayeredSpace(_PositionalAgentSpace):
    """
    LayeredSpace is a composite of _AbstractSpace's, each named.
    """
    def __init__(self, layers: Dict[str, _PositionalSpace] = {}, order: List[str] = []):
        super().__init__()
        self.layers: Dict[str, _PositionalSpace] = layers
        self.order: List[str] = order
        self._agent_to_layer: Dict[Agent, str] = {}

    def __getitem__(self, pos: Union[str, LayeredPosition]) -> Union[_AbstractSpace, Content]:
        if isinstance(pos, str):
            return self.layers[pos]
        else:
            return self.layers[pos.layer][pos.pos]

    def __setitem__(self, pos: LayeredPosition, content: Content) -> None:
        self.layers[pos.layer][pos.pos] = content
        if isinstance(content, Agent):
            self._agent_to_layer[content] = pos.layer

    def __delitem__(self, pos_or_content: Union[LayeredPosition, Content]) -> None:
        if isinstance(pos_or_content, LayeredPosition):
            del self.layers[pos_or_content.layer][pos_or_content.pos]
        else:
            for l in self.layers.values():
                if isinstance(l, _PositionalAgentSpace) and pos_or_content in l:
                    del l[pos_or_content]

    def __contains__(self, pos_or_content: Union[str, LayeredPosition, Content]) -> bool:
        if isinstance(pos_or_content, LayeredPosition):
            return pos_or_content.pos in self.layers[pos_or_content.layer]
        elif isinstance(pos_or_content, str):
            return pos_or_content in self.layers
        else:
            for l in self.layers.values():
                if isinstance(l, _PositionalAgentSpace):
                    if pos_or_content in l:
                        return True

        return False

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
                    common_discreet_positions |= set(layer)
                elif len(common_discreet_positions) == 0:
                    break
                else:
                    common_discreet_positions &= set(layer)
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

    def place_agent(self, pos: LayeredPosition, agent: Agent) -> None:
        """Place an agent at a specific position."""
        if isinstance(self.layers[pos.layer], _PositionalAgentSpace):
            cast(_PositionalAgentSpace, self.layers[pos.layer]).place_agent(agent, pos.pos)
            self._agent_to_layer[agent] = pos.layer
        else:
            raise TypeError("Cannot add agent to layer '{}' because it is not of type _PositionalAgentSpace".format(pos.layer))

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""
        if isinstance(self.layers[self._agent_to_layer[agent]], _PositionalAgentSpace):
            cast(_PositionalAgentSpace, self.layers[self._agent_to_layer[agent]]).remove_agent(agent)
        else:
            raise TypeError("Something went wrong!  Cannot remove agent {} because \
                it is mapped to layer {}, which is not an _PositionalAgentSpace.  The layer \
                mapping has been removed.".format(agent, self._agent_to_layer[agent]))
        del self._agent_to_layer[agent]

    def agents_at(self, pos: Union[Position, LayeredPosition]) -> Iterator[Agent]:
        """Yield the agents at a specific position within a specific layer if
        the passed `pos` is of type `LayeredPosition`, else yield the agents at
        the passed `Position` for all layers"""
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _PositionalAgentSpace):
                return cast(_PositionalAgentSpace, self.layers[pos.layer]).agents_at(pos.pos)
            else:
                raise TypeError("Cannot return agents from layer '{}', it is not \
                    an instance of _PositionalAgentSpace.".format(pos.layer))
        else:
            return chain(*[l.agents_at(pos) for l in self.layers.values() if isinstance(l, _PositionalAgentSpace)])

    def count_agents_at(self, pos: Union[Position, LayeredPosition]) -> int:
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _PositionalAgentSpace):
                return cast(_PositionalAgentSpace, self.layers[pos.layer]).count_agents_at(pos.pos)
            else:
                raise TypeError("Cannot return agents from layer '{}', it is not \
                    an instance of _PositionalAgentSpace.".format(pos.layer))
        else:
            return sum([l.count_agents_at(pos) for l in self.layers.values() if isinstance(l, _PositionalAgentSpace)])

    def neighbors_at(self, pos: Union[Position, LayeredPosition], radius: Distance = 1, include_center: bool = True) -> Iterator[Tuple[Position, Agent]]:
        """Yield the agents in proximity to a position."""
        if isinstance(pos, LayeredPosition):
            if isinstance(self.layers[pos.layer], _PositionalAgentSpace):
                return cast(_PositionalAgentSpace, self.layers[pos.layer]).neighbors_at(pos.pos, radius, include_center)
            else:
                raise TypeError("Cannot get neighbors at pos '{}' in layer '{}' \
                    because it is not of type _PositionalAgentSpace".format(pos.pos, pos.layer))
        else:
            return chain(*[l.neighbors_at(pos, radius, include_center) for l in self.layers.values() if isinstance(l, _PositionalAgentSpace)])

    def neighbors_of(self, agent: Agent, radius: Distance = 1, include_center: bool = True) -> Iterator[Tuple[Position, Agent]]:
        """Yield the neighbors of an agent."""
        return cast(_PositionalAgentSpace, self.layers[self._agent_to_layer[agent]]).neighbors_of(agent, radius, include_center)

    def neighborhood_of(self, agent: Agent, radius: Distance = 1, include_center: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        """Yield the neighborhood of an agent."""
        return cast(_PositionalAgentSpace, self.layers[self._agent_to_layer[agent]]).neighborhood_of(agent, radius, include_center)


class EuclidianGridMetric(_Metric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(EuclidianGridMetric, cls).distance(coord1, coord2, space)

        return sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

    @classmethod
    def path(cls, pos1: Position, pos2: Position, space: _AbstractSpace) -> Iterator[Position]:
        return iter([pos1, pos2])

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace, include_center: bool = True) -> Iterator[GridCoordinate]:
        super(EuclidianGridMetric, cls).neighborhood(center, radius, space)

        # This is ugly and inefficient, but this will grind out the needed result
        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                if cls.distance((0, 0), (x, y), space) <= radius and (include_center or (x != 0 and y != 0)):
                    yield (center[0]+x, center[1]+y)


class ManhattanGridMetric(_Metric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(ManhattanGridMetric, cls).distance(coord1, coord2, space)

        return abs(coord1[0]-coord2[0]) + abs((coord1[1]-coord2[1]))

    @classmethod
    def path(cls, pos1: Position, pos2: Position, space: _AbstractSpace) -> Iterator[Position]:
        return iter([pos1, pos2])

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace, include_center: bool = True) -> Iterator[Position]:
        super(ManhattanGridMetric, cls).neighborhood(center, radius, space)

        for y in range(-int(radius), int(radius)+1):
            for x in range(abs(y)-int(radius), int(radius)-abs(y)+1):
                if include_center or (x != 0 and y != 0):
                    yield (center[0]+x, center[1]+y)


class ChebyshevGridMetric(_Metric):
    @classmethod
    def distance(cls, coord1: GridCoordinate, coord2: GridCoordinate, space: _AbstractSpace) -> Distance:
        super(ChebyshevGridMetric, cls).distance(coord1, coord2, space)

        return max(abs(coord1[0]-coord2[0]), abs((coord1[1]-coord2[1])))

    @classmethod
    def path(cls, pos1: Position, pos2: Position, space: _AbstractSpace) -> Iterator[Position]:
        return iter([pos1, pos2])

    @classmethod
    def neighborhood(cls, center: GridCoordinate, radius: Distance, space: _AbstractSpace, include_center: bool = True) -> Iterator[Position]:
        super(ChebyshevGridMetric, cls).neighborhood(center, radius, space)

        for y in range(-int(radius), int(radius)+1):
            for x in range(-int(radius), int(radius)+1):
                if include_center or (x != 0 and y != 0):
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
    def max1(space: _PositionalAgentSpace, coord: GridCoordinate, agent: Agent) -> bool:
        caller = ConsistencyChecks._get_caller()

        if caller == "__setitem__":
            val = space[coord]
            if isinstance(val, collections.Iterator):
                return not len(list(val))

        return True

    @staticmethod
    def unique(space: _PositionalAgentSpace, coord: GridCoordinate, agent: Agent) -> bool:
        caller = ConsistencyChecks._get_caller()

        if caller == "__setitem__":
            val = space[coord]
            if isinstance(val, collections.Iterator):
                return type(agent) not in map(type, space[coord])

        return True

# Need to create a e.g. Cartesian abstract class that is a space indexed by (x, y)???
# class _Grid()...
class _Grid(_PositionalSpace):
    def __init__(self, width: int, height: int, torus: bool, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.torus = torus

    def _verify_coord(self, pos: GridCoordinate, raise_exception: bool = True) -> Optional[GridCoordinate]:
        if 0 <= pos[0] <= self.width and 0 <= pos[1] < self.height:
            return pos
        elif self.torus:
            return (pos[0] % self.width, pos[1] % self.height)
        elif raise_exception:
            raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

        return None

class Grid(_PositionalAgentSpace, _Grid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._grid: Dict[GridCoordinate, Set] = dict()

    @property
    def default_value(self) -> Set:
        return set()

    @property
    def is_continuous(self) -> bool:
        return False

    def __getitem__(self, pos: GridCoordinate) -> set:
        return self._grid.get(cast(GridCoordinate, self._verify_coord(pos)), self.default_value)

    def __setitem__(self, pos: GridCoordinate, agent: Agent) -> None:
        pos = cast(GridCoordinate, self._verify_coord(pos))

        if self.consistency_check is not None and self.consistency_check(self, pos, agent):
            try:
                self._grid[pos].add(agent)
                self._agent_to_pos[agent] = pos
            except KeyError:
                self._grid[pos] = set([agent, ])
                self._agent_to_pos[agent] = pos
        else:
            raise ValueError("Cannot add agent {} to position {}, failed consistency check {}".format(agent, pos, self.consistency_check))

    def __delitem__(self, pos_or_content: Union[GridCoordinate, Agent]) -> None:
        if isinstance(pos_or_content, tuple):
            pos = cast(GridCoordinate, self._verify_coord(pos))
            for a in self._grid[pos]:
                del self._agent_to_pos[a]
            self._grid[pos].clear()
        else:
            self._grid[self._agent_to_pos[pos_or_content]].remove(pos_or_content)
            del self._agent_to_pos[pos_or_content]

    def __contains__(self, pos_or_content: Union[GridCoordinate, Agent]) -> bool:
        if isinstance(pos_or_content, tuple):
            return self._verify_coord(pos_or_content, False) is not None
        else:
            return pos_or_content in self._agent_to_pos

    def __missing__(self, pos: GridCoordinate) -> Set:
        if pos in self:
            return self.default_value

        raise LookupError("'{}' is out of bounds for width of {} and height of {}".format(pos, self.width, self.height))

    def neighborhood_at(self, pos: GridCoordinate, radius: Distance = 1, include_center: bool = True) -> Iterator[GridCoordinate]:
        pos = cast(GridCoordinate, self._verify_coord(pos))

        return cast(Iterator[GridCoordinate], self.metric.neighborhood(pos, radius, self, include_center))

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        return iter(self[pos])

    def count_agents_at(self, pos: Position) -> int:
        return len(self[pos])

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Iterator[Tuple[GridCoordinate, Agent]]:
        return chain(*[map(lambda a: (pos, a), self.agents_at(pos)) for n in self.neighborhood_at(pos, radius, include_center)])

    def get_all_positions(self) -> Iterator[Position]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)


class NetworkXMetric(_Metric):
    @classmethod
    def distance(cls, node1: Position, node2: Position, space: _AbstractSpace) -> Distance:
        super(NetworkXMetric, cls).distance(node1, node2, space)

        return nx.shortest_path_length(cast(PositionalAgentNetworkX, space)._graph, node1, node2, weight="distance")

    @classmethod
    def neighborhood(cls, root: Position, radius: Distance, space: _AbstractSpace, include_center: bool = True) -> Iterator[Position]:
        super(NetworkXMetric, cls).neighborhood(root, radius, space)

        return ego_graph(cast(PositionalAgentNetworkX, space)._graph, root, radius, center=include_center, distance="distance").nodes


class PositionalAgentNetworkX(_PositionalAgentSpace):
    def __init__(self,
                graph: Optional[nx.Graph] = None,
                **kwargs):
        super().__init__(**kwargs)
        if graph is None:
            self._graph = nx.Graph()
        else:
            graph_copy = copy.copy(graph)
            self._verify_entire_graph(graph_copy)
            self._graph = graph_copy


    def _verify_entire_graph(self, graph: nx.Graph) -> None:
        for pos, attr in graph.nodes(data="agents"):
            if attr is None:
                attr = self.default_value
                graph.nodes[pos]["agents"] = attr
            if not isinstance(attr, set):
                raise TypeError("All nodes of passed graph must have attribute 'agents' be of type set (or None); found value {} of type {}".format(attr, type(attr)))

            for a in attr:
                if not isinstance(a, Agent):
                    raise TypeError("All contens of NetworkX nodes must be of type Agent; found {}".format(a))

    @property
    def is_continuous(self) -> bool:
        return False

    @property
    def default_value(self) -> Content:
        return set()

    def __contains__(self, pos_or_content: Union[Position, Content]) -> bool:
        if isinstance(pos_or_content, Agent):
            return pos_or_content in self._agent_to_pos
        else:
            return pos_or_content in self._graph

    def get_all_positions(self) -> Iterator[Position]:
        return self._graph.nodes

    def neighborhood_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Union[Iterator[Position], _PositionalSpace]:
        if pos not in self._graph:
            raise LookupError("'{}' is not part of the network graph".format(pos))

        return cast(Iterator[Position], self.metric.neighborhood(pos, radius, self, include_center))

    def agents_at(self, pos: Position) -> Iterator[Agent]:
        return iter(self[pos])

    def count_agents_at(self, pos: Position) -> int:
        return len(self[pos])

    def neighbors_at(self, pos: Position, radius: Distance = 1, include_center: bool = True) -> Iterator[Tuple[Position, Content]]:
        return chain(*[map(lambda a: (npos, a), self.agents_at(npos)) for npos in cast(Iterator[Any], self.neighborhood_at(pos, radius, include_center))])

    def __getitem__(self, pos: Position) -> set:
        return self._graph.nodes[pos]["agents"]

    def __setitem__(self, pos: Position, agent: Agent) -> None:
        if pos not in self._graph:
            raise LookupError("'{}' is not part of the network graph".format(pos))

        self._graph.nodes[pos]["agents"].add(agent)
        self._agent_to_pos[agent] = pos

    def __delitem__(self, pos_or_content: Union[Position, Agent]) -> None:
        if pos_or_content not in self:
            raise LookupError("'{}' is not part of the network graph".format(pos_or_content))

        if isinstance(pos_or_content, Agent):
            self._graph.nodes[self.find_agent(pos_or_content)]["agents"].remove(pos_or_content)
        else:
            for a in self._graph.nodes[pos_or_content]["agents"]:
                del self._agent_to_pos[a]
            self._graph.nodes[pos_or_content]["agents"].clear()

    def __missing__(self, pos: Position) -> Content:
        raise LookupError("'{}' is not part of the network graph".format(pos))


class NumpyPatchConsistencyChecks(ConsistencyChecks):
    @staticmethod
    def gteq0(space: NumpyPatchGrid, coord: Optional[GridCoordinate], value: Optional[Content]) -> Union[bool, np.ndarray]:
        if coord is None:
            return space._grid >= 0

        caller = ConsistencyChecks._get_caller()
        if caller == "__setitem__":
            if value is not None:
                return value >= 0
            else:
                return False

        return True


# An altrnative path would be to have NumpyPatchGrid actually extend numpy.ndarray
# but I have some doubts. E.g. implementing consistency checks would require wrapping
# most methods, less so but still with torus, etc.  That is all "fine", though it
# doesn't save work.  Worse, would have to disable e.g. being able to reshape,
# in-place sort, and changing of various lower-level flags etc.  But it would be
# neat if we just extended it...
class NumpyPatchGrid(_PositionalPatchSpace, _Grid):
    def __init__(self, init_val: np.ndarray, **kwargs):
        _grid = np.array(init_val)  # We don't need no stinkin' matrices here
        _height, _width = _grid.shape
        super().__init__(height=_height, width=_width, **kwargs)
        if _grid.ndim != 2:
            raise TypeError("NumericPatchGrid may only be initilialized with a ndarray of dimension 2")
        self._grid = _grid

    def _verify_other(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> Union[np.ndarray, int, float]:
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

    def __iadd__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid += self._verify_other(other)
        return self

    def __isub__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid -= self._verify_other(other)
        return self

    def __imul__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid *= self._verify_other(other)
        return self

    def __itruediv__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid /= self._verify_other(other)
        return self

    def __ipow__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid **= self._verify_other(other)
        return self

    def __iand__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid &= self._verify_other(other)
        return self

    def __ior__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid |= self._verify_other(other)
        return self

    def __ixor__(self, other: Union[NumpyPatchGrid, np.ndarray, int, float]) -> NumpyPatchGrid:
        self._grid ^= self._verify_other(other)
        return self

    @property
    def default_value(self) -> Optional[Content]:
        return None

    def content(self) -> Iterator[Content]:
        return iter(self._grid)

    def all(self) -> Iterator[Tuple[Position, Optional[Content]]]:
        return np.ndenumerate(self._grid)

    def __getitem__(self, pos: GridCoordinate) -> Content:
        return self._grid[pos]

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
