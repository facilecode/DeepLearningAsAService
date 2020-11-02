import importlib.util

spec = importlib.util.spec_from_file_location("module.name", "tf_module.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

print(dir(foo))