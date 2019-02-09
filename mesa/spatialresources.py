from .space import Grid

class PropertyGrid(Grid):
    def __init__(self, width, height, torus, allowed_property_types):
        """ Create a new Property grid.  A grid that has zero or more static
        properties associated with each grid cell.  E.g. 'elevation', expressed
        as some fixed number, or 'rainfall' expressed as a callable which takes
        position and time.

        Args:
            width, height: The width and width of the grid
            torus: Boolean whether the grid wraps or not.
            allowed_property_types:  iterable of allowed property types (name)
        """
        super().__init__(width, height, torus)
        self.allowed_property_types = allowed_property_types
        self.properties = {}

    def add_property_layer(self, property, values):
        if len(values) == len(self.grid) and len(values[0]) == len(self.grid[0]):
            self.properties[property] = values

    def remove_property_layer(self, property):
        del self.properties[property]

    def get_property(self, property, pos):
        if property in self.properties:
            x, y = pos
            return self.properties[property][x][y]

        return None
