from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterator, List, Type

from .constants import DATA_SOURCES_DIR, MODULE_SUFFIX

from space_weather_api import SpaceWeatherAPI


def iter_data_source_modules() -> Iterator[str]:
    prefix = "data_sources."
    for module in pkgutil.walk_packages([str(DATA_SOURCES_DIR)], prefix=prefix):
        if module.ispkg:
            continue
        if module.name.split(".")[-1].endswith(MODULE_SUFFIX):
            yield module.name


def load_data_source_classes() -> List[Type[SpaceWeatherAPI]]:
    classes: List[Type[SpaceWeatherAPI]] = []
    for module_name in iter_data_source_modules():
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, SpaceWeatherAPI) and obj is not SpaceWeatherAPI:
                classes.append(obj)
    classes.sort(key=lambda cls: cls.__name__)
    return classes


__all__ = ["iter_data_source_modules", "load_data_source_classes"]
