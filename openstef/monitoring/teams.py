# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

import pandas as pd
import pymsteams
import structlog


def post_teams(
    msg: str,
    invalid_coefs: pd.DataFrame = None,
    coefsdf: pd.DataFrame = None,
    url: str = None,
    proxies: dict = None,
) -> None:
    """Post a message to Teams - KTP

    Note that currently no authentication occurs.
    Security is given by keeping the URL secret.
    One should therefore refrain from using more enhanced features such as
    action buttons.

    Args:
        msg (mixed): For simple messages a string can be passed. For more
            complex messages pass a dict. The following keys are supported:
            text, links, sections. Each section can contain the following keys:
            text, title, images, facts, markdown. Also see:
            https://docs.microsoft.com/en-us/outlook/actionable-messages/send-via-connectors
        invalid_coefs (pd.DatFrame, optional): df of information of invalid
            coefficients. Defaults to None.
        coefsdf (pd.DataFrame, optional): df of new coefficients. Defaults to None.
        url (string, optional): webhook url, monitoring by default

    Note:
        This function is namespace-specific.
    """
    logger = structlog.get_logger(__name__)

    # Add invalid coefficients and manual coefficients-query to message
    if invalid_coefs is not None and coefsdf is not None:
        # add invalid coefficient information to message in dict-format
        invalid_coefs_text = "".join(
            [
                f"\n* **{row.coef_name}**: {round(row.coef_value_new, 2)}, "
                f"(previous: {round(row.coef_value_last, 2)})"
                for index, row in invalid_coefs.iterrows()
            ]
        )
        query = build_sql_query_string(coefsdf, "energy_split_coefs")
        query_text = (
            "If you would like to update the coefficients manually in the "
            + "database, use this query:"
        )
        msg = {
            "fallback": msg,
            "title": "Invalid energy splitting coefficients",
            "text": msg,
            "sections": [
                {
                    "text": invalid_coefs_text,
                    "markdown": True,
                },
                {
                    "title": "Manual query",
                    "text": query_text,
                    "markdown": True,
                },
                {
                    "text": query,
                    "markdown": True,
                },
            ],
        }

    # If no url is passed fall back to default
    if url is None:
        # if Teams url is not context.configured just return
        if url == None:
            logger.warning("Can't post Teams message, no url given")
            return

    card = pymsteams.connectorcard(url)

    # add proxies
    # NOTE the connectorcard.proxy is passed to the requests library under the hood
    card.proxies = proxies

    # if msg is string, convert to dict
    if type(msg) is str:
        msg = dict(text=msg)
    card.text(msg.get("text"))
    card.summary(msg.get("fallback", "-"))

    # set title, color, ...
    card.color(msg.get("color", "white"))
    card.title(msg.get("title"))

    link_dicts = msg.get("links", [])  # link_dicts can be single dict or list of dicts
    if isinstance(link_dicts, dict):  # if single dict
        card.addLinkButton(link_dicts["buttontext"], link_dicts["buttonurl"])
    elif isinstance(link_dicts, list):  # if list of dicts
        for link_dict in link_dicts:
            card.addLinkButton(link_dict["buttontext"], link_dict["buttonurl"])

    # Add sections
    for section_dict in msg.get("sections", []):
        section = pymsteams.cardsection()

        section.text(section_dict.get("text"))
        section.title(section_dict.get("title"))
        for image in section_dict.get("images", []):
            section.addImage(image)
        for fact in section_dict.get("facts", []):
            section.addFact(*fact)
        if not section_dict.get("markdown", True):
            section.disableMarkdown()
        if "link" in section_dict:
            section.linkButton(
                section_dict.get("link").get("buttontext"),
                section_dict.get("link").get("buttonurl"),
            )

        card.addSection(section)

    card.send()


def build_sql_query_string(df, table):
    """Build sql insert query string for Teams message output from df.

    Args:
        df (pd.DataFrame): df of table values to insert in sql
        table (string): table to insert df into

    Returns:
        string: sql query string of insert statement
    """
    # round all values to two decimals
    df = df.round(2)
    # convert datetime to string format
    datetime_columns = df.columns[
        df.columns.isin(["date_start", "date_end", "created"])
    ]
    for col in datetime_columns:
        df[col] = df[col].astype("str")

    sql_texts = [
        "```  \nINSERT INTO "
        + table
        + " ("
        + str(", ".join(df.columns))
        + ") VALUES  \n"
    ]
    for index, row in df.iterrows():
        if index != df.index[0]:
            sql_texts.append(",  \n")  # 2 spaces and \n create a new line
        sql_texts.append(str(tuple(row.values)))
    sql_texts.append("  \n```")
    query = "".join(sql_texts)
    return query


def format_message(title: str, params: dict, fallback=None, color=None) -> dict:
    if color is None:
        color = "#046b00"  # green
    if fallback is None:
        fallback = title

    # format allparams using limited precision for floats
    # make keys bold (**key**)
    values = []
    for k, v in params.items():
        if type(v) is float:
            values.append(f"**{k}**: {v:0.3f}")
            continue
        values.append(f"**{k}**: {v}")
    # join all {params}: {value}  pairs with a new line
    text = "".join([f"* {v}   \n" for v in values])

    msg = {
        "fallback": fallback,
        "title": title,
        "text": text,
        "color": color,
    }
    return msg
