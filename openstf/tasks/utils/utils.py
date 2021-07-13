# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import json
import os
from pathlib import Path

from openstf_dbc.config.config import ConfigManager


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
