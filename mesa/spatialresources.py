from functools import partial

class PropertyGrid:
    def __init__(self, torus, property_name, grid):
        self.width = grid.shape[0]
        self.height = grid.shape[1]
        self.torus = torus
        self.property_name = property_name
        self.grid = np.copy(grid)

    @property
    def property_type(self):
        return self.grid.dtype

    def get_property_at(self, pos):
        return self.property_name, self.grid[pos]


class FunctionPropertyGrid(PropertyGrid):
    def __init__(self, height, width, torus, property_name, function):
        """ Create a new Property grid.  A grid that has zero or more simple
        properties associated with each grid cell.  E.g. 'elevation', expressed
        as some fixed number, or 'rainfall' expressed as a callable which takes
        position and time.  Note that callable is stored per-location, so is not
        necessarily the same as a single function that could take the grid as an
        argument and perform some simple numpy fucntions on the whole grid.

        Args:
            width, height: The width and width of the grid
            torus: Boolean whether the grid wraps or not.
            allowed_property_types:  iterable of allowed property types (name)
        """
        super().__init__(torus, property_name, function(0, (width, height)))
        self.width = width
        self.height = height
        self.function = function

    def step(self, time):
        self.grid = function(self.width, self.height, time)


class IrregularFunctionPropertyGrid(PropertyGrid):
    @staticmethod
    def _call_grid(functions_grid, time):
        ret = np.empty(functions_grid.shape, dtype=type(functions_grid[0, 0](0, (0, 0) )) )
        for index in np.ndindex(*functions_grid.shape):
            ret[index] = functions_grid[index](time, index)
        return ret

    def __init__(self, height, width, torus, property_type, functions_grid):
        """ Create a new Property grid.  A grid that has zero or more simple
        properties associated with each grid cell.  E.g. 'elevation', expressed
        as some fixed number, or 'rainfall' expressed as a callable which takes
        position and time.  Note that callable is stored per-location, so is not
        necessarily the same as a single function that could take the grid as an
        argument and perform some simple numpy fucntions on the whole grid.

        Args:
            width, height: The width and width of the grid
            torus: Boolean whether the grid wraps or not.
            allowed_property_types:  iterable of allowed property types (name)
        """
        super().__init__(torus, property_name, _call_grid(functions_grid, 0))
        self.width = width
        self.height = height
        self.functions_grid = functions_grid

    def step(self, time):
        self.grid = _call_grid(self.functions_grid, time)
