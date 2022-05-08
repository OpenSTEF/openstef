# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501
#
# SPDX-License-Identifier: MPL-2.0

import re

with open("setup.py", "r") as file:
    version_content = file.read()

# Match regex for version=0.0.0a, pattern
minor_version_number_with_suffix = re.findall(
    r'version="\d+\.\d+\.([a-zA-Z0-9]+)",', version_content
)[0]
minior_version_number = int(re.findall(r"\d+", minor_version_number_with_suffix)[0])
regex_bumped_minor_version_number = f"\g<1>{minior_version_number + 1}"

# Match regex for version=0.0.0a pattern
bumped_version_content = re.sub(
    r'(version="\d+\.\d+\.)[a-zA-Z0-9]+',
    regex_bumped_minor_version_number,
    version_content,
)

with open("setup.py", "w") as file:
    file.write(bumped_version_content)
