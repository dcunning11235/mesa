class ConsistencyChecks:
    """ConsistencyChecks come in two varieties:  vetoing and clamping.  Vetoing
    checks return a bool as to whether the change was allowed to happen.  Clamping
    checks return the result that was allowed."""
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

    @staticmethod
    def get_arb_test(values: np.ndarray, func: Callable[Union[np.ndarray, Content], Union[np.ndarray, Content]]) -> Union[np.ndarray, Content]:
        class NpArbTest(ConsistencyChecks):
            def __init__(values: np.ndarray):
                self.values: np.ndarray = values

            def test(space: NumpyPatchGrid, coord: Optional[GridCoordinate], value: Optional[Content]) -> Union[np.ndarray, Content]:
                if value is None:
                    if coord is None:
                        return func(space._grid, self.values)
                    else:
                        return func(space._grid[coord], self.values[coord])
                else:
                    caller = ConsistencyChecks._get_caller()
                    if caller == "__setitem__" and coord is not None:
                        return func(value, self.values[coord])


    @staticmethod
    def lteq_arb(space: NumpyPatchGrid, coord: Optional[GridCoordinate], value: Optional[Content]) -> Union[bool, np.ndarray]:
        if coord is None:
            return space._grid >= 0

        caller = ConsistencyChecks._get_caller()
        if caller == "__setitem__":
            if value is not None:
                return value >= 0
            else:
                return False

        return True
