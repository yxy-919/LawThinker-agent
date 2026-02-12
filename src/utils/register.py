import os
import sys
import importlib
from types import ModuleType

project_root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, alias, class_reference):
        self._registry[alias] = class_reference

    # 根据别名动态获取类；若尚未注册，尝试按约定路径导入对应模块
    def get_class(self, alias):
        # 已注册直接返回
        cls = self._registry.get(alias)
        if cls is not None:
            return cls

        # 尝试按 "Group.ClassName" 约定动态导入
        # e.g. "Engine.qwen3_8B" -> "src.engine.qwen3_8B"
        if '.' in alias:
            group, name = alias.split('.', 1)
            module_path = f"src.{group.lower()}.{name}"
            try:
                # 若重复导入不会有副作用
                importlib.import_module(module_path)
                return self._registry.get(alias)
            except ModuleNotFoundError:
                # 最后再尝试大小写保持一致
                try:
                    importlib.import_module(f"src.{group}.{name}")
                    return self._registry.get(alias)
                except ModuleNotFoundError:
                    pass
        return None


registry = Registry()

def register_class(alias=None):
    def decorator(cls):
        nonlocal alias
        if alias is None:
            alias = cls.__name__
        registry.register(alias, cls)
        return cls
    return decorator
