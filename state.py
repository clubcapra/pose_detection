class StateType(enumerate):
    IDLE = 0
    FOLLOW = 1
    RETRACE = 2

class State:
    def __init__(self):
        self.state = StateType.IDLE

    def setState(self, state: StateType) -> None:
        self.state = state

    def getState(self) -> StateType:
        return self.state

    