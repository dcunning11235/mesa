from abc import ABC
from mesa.agent import Agent
from typing import Any, Tuple, Set

Position = Any
Content = Any

GridCoordinate = Tuple[int, int]
CellContent = Set[Any]

LayeredPosition = Tuple[string, Position]
LayeredContent = Dict[str, Content]

'''
Other things _AbstractSpace and subclasses might have:
    Grid:  Max occupancy (1, inf, 7, etc.)
    LayeredSpace:  Max occupancy (1, inf, 7, etc.) by some type (e.g. 2 Agents
                    are allowed across all layers, but maybe inf content.)
    NOTE: Maybe the above two point to a need for some general "consistency checks"
        functionality, e.g. a function that the user can pass in to a space that
        is called whenever a cell/node/etc. has its contents (Agents, Contents,
        etc.) updated.  The function would have a signature like:
        consisitency_check(space, position) -> bool where a False vetos the change
        and a True allows it.
'''
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


class _PatchSpace(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    


class LayeredSpace(_AbstractSpace):
    def __init__(self):
        super().__init__()
        self.layers: Dict[str, _AbstractSpace] = {}

    def __getitem__(self, pos: LayeredPosition) -> Content:
        #Relies on __getitem__  on _AbstractSpace implementations not throwing
        #   KeyError's but returning defaults, BUT ALSO doing their own bounds
        #   checking and throwing errors appropriately.
        return self.layers[pos[0]][pos[1]]

    def __setitem__(self, pos: LayeredPosition, content: Content) -> None:
        self.layers[pos[0]][pos[1]] = content

    def __delitem__(self, content: Tuple[LayeredPosition, Content]) -> None:
        self.layers[content[0][0]].__delitem__( (content[0][1], content[1]) )


    def place_agent(self, agent: Agent, pos: LayeredPosition) -> None:
        """Place an agent at a specific position."""

        self[pos] = agent
        #Tricky.  Since e.g. "Grid" doesn't know anything about layers, have to
        #store the GridCoordinate in "pos", not the LayeredPosition.
        #Should setattr "layer"?
        setattr(agent, "layer", pos[0])
        setattr(agent, "pos", pos[1])

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space."""

        old_pos = getattr(agent, "pos")
        old_layer = getattr(agent, "layer")

        del self[(old_layer, old_pos), agent]
        setattr(agent, "layer", None)
        setattr(agent, "pos", None)

    def move_agent(self, agent: Agent, pos: LayeredPosition) -> None:
        """Move an agent from its current to a new position."""

        old_pos = getattr(agent, "pos")
        older_layer = getattr(agent, "layer")

        del self[(old_layer, old_pos), agent]
        self[pos] = agent
        setattr(agent, "layer", pos[0])
        setattr(agent, "pos", pos[1])

    '''
    So now the question is, what "stacked" methods should be added and how should
    returned values look?  E.g. def agents_at(self, pos: Position) returns
    all the agents from all the layers as a single iterator?  A dict of
    "layername":_AbstractSpace key:values?

    It makes sense to group layers into (at least) Content and Agent layers... I
    think...  Should layers have a type associated with them, too?  No, right? I
    think that is a bad idea.  (You should have to specify the layer you want to
    e.g. place an Agent into, not be able to rely on type-based placement...)

    How would different spaces stack together?  How would a grid and network stack?
    Are there any special considerations there, anything we want to make easy?
    ...Or should layers all be required to be of the same type?  I'm trying to
    think of a not-completely-contrived example of where this would make sense.
    Maybe something like a (Multi)Grid where free-moving agents can coalesce into
    "cities" that then have NetworkGrid connections between them.  Okay, so then
    the GridCoordinate of a city (e.g. (456, 789)) would have to be the node ID/
    position/label for the NetworkGrid.  Not too clunky, not too bad.  Does speak
    to the potential usefulness of being able to attach attributes to locations
    in space:  so now we have Agents, Content, and (a/A)ttributes.
    '''


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
