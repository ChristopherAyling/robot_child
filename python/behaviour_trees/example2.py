import py_trees
from py_trees.common import Status
import random

nums = [2, 3, 0, 8]

class SortedCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)

    def setup(self, **kwargs):
        pass

    def initialise(self) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        if sorted(nums) == nums:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status: Status) -> None:
        pass

class SortAction(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super().__init__(name)

    def setup(self, **kwargs):
        pass

    def initialise(self) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        random.shuffle(nums)
        if sorted(nums) == nums:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        pass

condition = SortedCondition("is_sorted")
action = SortAction("sort")

root = py_trees.composites.Selector(name="Root", memory=True)
root.add_children([condition, action])

root.setup_with_descendants()


for i in range(30):
    print(py_trees.display.unicode_tree(root, show_status=True))
    root.tick_once()
    print(i)
