# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path

from ktpbase.config.config import ConfigManager


def load_status_file():
    """
    This function loads the json status file
    :return: dictionary (status object)
    """
    status_path = Path(ConfigManager.get_instance().paths.webroot)
    status_file = "status.json"
    # Check if current status exists
    if os.path.isfile(status_path / status_file):
        # Load current status
        status_file = open(status_path / status_file, "r")
        status_file_string = status_file.read()
        try:
            status_dict = json.loads(status_file_string)
        except json.JSONDecodeError:
            status_file_string = status_file_string.replace("'", '"')
            status_dict = json.loads(status_file_string)
    else:
        # Create empty status
        status_dict = {}
    return status_dict


def write_status_file(status):
    """
    This function writes the given status object to a json file
    :param status: dictionary
    :return: None
    """
    status_path = Path(ConfigManager.get_instance().paths.webroot)
    status_file = "status.json"

    # If path does not exists, create it
    if not os.path.exists(status_path):
        os.makedirs(status_path)

    # Write status file
    with open(status_path / status_file, "w") as json_file:
        json.dump(status, json_file)


def check_status_change(name, new_code):
    """
    Check if the old status code equals the new code.
    If no status if found it is assumed that the old code is 0
    :param name: name of the status field
    :param new_code: int
    :return: True if changed
    """

    # Load previous status
    status = load_status_file()

    # Get old status
    old_code = 0
    if name in status:
        old_code = status[name]

    # Equality check
    return old_code != new_code


def update_status_change(name, new_code):
    """

    :param name: name of the status field
    :param new_code: int
    :return: None
    """

    # Load status
    status = load_status_file()

    # Update status
    status[name] = new_code

    # Write status
    write_status_file(status)


def convert_string_args_to_dict_args(argsstring):
    """Helper function that converts a string of arguments to a dict of arguments.
    Coverts inputs to floats if possible. Assumes args are seperated by ',' or ', '.

    input: str

    output: dict(args=[arg1, ..., argN], kwargs=dict(kwarg1=val1, ..., kwargN=valN)

    example:
        '5, seconds' -> dict(args=[5, 'seconds'])
        '3, val=blub -> dict(args=[3], kwargs=dict(val='blub')"""

    # convert ', 'to ',' and split args on ','
    args = argsstring.replace(", ", ",").split(",")
    # Make list of all items without a '=' -> the args
    arg_list = [x for x in args if "=" not in x]

    # Make a list of all items with a '=' -> the kwargs
    # kwarglist should be like ['key1=val1','key2 = val2']
    kwarg_list = [x for x in args if x not in arg_list]
    # Now let's turn the list into a dict. e.g. {val1:'key1', val2:'key2'}
    kwargs_dict = {
        key: value
        for key, value in [
            kwarg_pair.replace(" = ", "=").split("=") for kwarg_pair in kwarg_list
        ]
    }

    # Lastly, convert values to floats if possible
    arg_list = [float(x) if x.replace(".", "").isdigit() else x for x in arg_list]
    kwargs_dict = {
        key: float(value) if value.replace(".", "").isdigit() else value
        for key, value in kwargs_dict.items()
    }

    return dict(args=arg_list, kwargs=kwargs_dict)


def interpret_string_as_functions(function, args, kwargs):
    """Interpret a given function with args and kwargs to build a string suitable for eval
    input:
        function = str
        args = list(arg1, ..., argN)
        kwargs = dict(kwarg1=val1, ..., kwargN=valN)

    return:
        function(arg1, ..., argN, kwarg1=val1, ..., kwargN=valN)
    """
    arg_string = ", ".join(
        [str(x) if type(x) in [float, int] else "'{}'".format(x) for x in args]
    )
    if arg_string == "":
        arg_string = arg_string + "db"
    else:
        arg_string = arg_string + ", db"

    kwarg_string = ""
    # build sorted list of kwarg keys; otherwise not able to unittest
    kwargkeys = sorted(kwargs.keys())
    for key in kwargkeys:
        value = kwargs[key]
        # add quotes if value is a string
        if not isinstance(value, (int, float)):
            value = "'" + value + "'"
        kwarg_string += "{}={}, ".format(key, value)
    # remove last ,
    if len(kwarg_string) > 2:
        kwarg_string = kwarg_string[:-2]

    # add comma between argstring and kwargstring if argstring not empty and kwargstring not empty
    if (len(arg_string) > 0) and (len(kwarg_string) > 0):
        arg_string += ", "

    # Build result
    result = function + "(" + arg_string + kwarg_string + ")"
    return result
