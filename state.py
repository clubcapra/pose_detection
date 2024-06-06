class StateType(enumerate):
    IDLE = 0
    FOLLOW = 1
    RETRACE = 2

class SystemState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.state = StateType.IDLE
        return cls._instance

    def set_state(self, new_state):
        if new_state in StateType:
            self.state = new_state
        else:
            raise ValueError("Invalid state")
    