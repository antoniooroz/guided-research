from enum import Enum
from typing import Any

def _parse_dict(default_dictionary: dict[Any, Any], given_dictionary: dict[str, Any]):
    key_list = list(default_dictionary)
    val_list = list(default_dictionary.values())
    
    if len(key_list) > 0:
        keyEnumClass = key_list[0].__class__
        keyIsEnum = isinstance(key_list[0], Enum)
    else:
        keyIsEnum = False
    
    if len(val_list) > 0:
        valEnumClass = val_list[0].__class__
        valIsEnum = isinstance(val_list[0], Enum)
    else:
        valIsEnum = False
    
    res = {}
    for key, val in given_dictionary.items():
        if keyIsEnum:
            key = keyEnumClass[key]
        if valIsEnum:
            val = valEnumClass[val]
            
        res[key] = val

    return res
        
def _parse_list(default_list: list[Any], given_list: list[Any]):
    if default_list is None:
        enumClass = None
        isEnum = False
    else:
        enumClass = default_list[0].__class__
        isEnum = isinstance(default_list[0], Enum)

    res = []
    for val in given_list:
        if isEnum:
            val = enumClass[val]
        
        res.append(val)    
        
    return res

def _parse_element(default_element: Any, given_element: Any):
    if isinstance(default_element, Enum):
        return default_element.__class__[given_element]
    else:
        return given_element

class BaseConfiguration():
    def to_dict(self, prefix='') -> dict[str, Any]:
        result: dict[str, Any] = {}
        
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            
            if isinstance(val, BaseConfiguration):
                result.update(val.to_dict(prefix=f'{prefix}{key}/'))
            elif isinstance(val, dict):
                new_val = {}
                for d_key, d_val in val.items():
                    if isinstance(d_key, Enum):
                        new_val[d_key.name] = d_val
                    else:
                        new_val[d_key] = d_val
                result[f"{prefix}{key}"] = new_val
            else:
                result[f"{prefix}{key}"] = val
                
        return result
    
    def from_dict(self, dictionary: dict[str, Any]):
        if dictionary is None:
            return
        
        for key, val in self.__dict__.items():
            if key in dictionary:
                if isinstance(val, BaseConfiguration):
                    self.__dict__[key] = val.__class__(dictionary[key])
                elif isinstance(val, dict):
                    self.__dict__[key] = _parse_dict(val, dictionary[key])
                elif isinstance(val, list):
                    self.__dict__[key] = _parse_list(val, dictionary[key])
                else:
                    self.__dict__[key] = _parse_element(val, dictionary[key])
