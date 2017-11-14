# -*- coding: utf-8 -*-

#版本检查
import sys
if sys.version > '3':
    PY3 = True
else:
    PY3 = False
    
#判断字典是否有键值
def dict_has_key( a_dict,a_key ):
    if PY3:
        return a_key in a_dict
    else:
        return a_dict.has_key( a_key )    