"""Maps mission JSON `action` strings to their handler class.

This is the whole plug-in mechanism: to add a new mission mode, write an
ActionHandler subclass under executor/actions/ and add one line here. The
dispatch loop in executor.executor.MissionExecutor never needs to change.

Kept in its own module (rather than inside executor.py) purely to avoid a
circular import: action modules need ExecutorContext from executor.executor,
and executor.py has no need to know about concrete action classes.
"""
from executor.actions.land import LandAction
from executor.actions.search_and_follow import SearchAndFollowAction
from executor.actions.takeoff import TakeoffAction

ACTION_REGISTRY = {
    "takeoff": TakeoffAction,
    "search_and_follow": SearchAndFollowAction,
    "land": LandAction,
}
