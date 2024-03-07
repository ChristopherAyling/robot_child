import py_trees

root = py_trees.composites.Sequence(name="Root", memory=True)

child1 = py_trees.behaviours.Success(name="Child1")
child2 = py_trees.behaviours.Success(name="Child2")
child3 = py_trees.behaviours.StatusQueue(
    name="Child3",
    queue=[
        py_trees.common.Status.FAILURE,
        py_trees.common.Status.FAILURE,
    ],
    eventually=py_trees.common.Status.SUCCESS,
)
child4 = py_trees.behaviours.Success(name="Child4")

root.add_children([child1, child2, child3, child4])

root.setup_with_descendants()

root.tick_once()
print(py_trees.display.unicode_tree(root, show_status=True))
root.tick_once()
print(py_trees.display.unicode_tree(root, show_status=True))
root.tick_once()
print(py_trees.display.unicode_tree(root, show_status=True))
root.tick_once()
print(py_trees.display.unicode_tree(root, show_status=True))
root.tick_once()
print(py_trees.display.unicode_tree(root, show_status=True))
