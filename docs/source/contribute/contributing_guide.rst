.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

.. _contributing_guide:

******************
Contributing guide
******************

You've discovered a bug or something else you want to change in OpenSTEF — excellent!

You've worked out a way to fix it — even better!

You want to tell us about it — best of all!

Below, you can find a number of ways to contribute, and how to connect with the
OpenSTEF community.

Ways to contribute
==================

.. dropdown:: Do I really have something to contribute to OpenSTEF?
    :open:
    :icon: person-fill

    100% yes! There are so many ways to contribute to our community. Take a look
    at the following sections to learn more.

    There are a few typical new contributor profiles:

    * **You are an OpenSTEF user, and you see a bug, a potential improvement, or
      something that annoys you, and you can fix it.**

      You can search our issue tracker for an existing issue that describes your problem or
      open a new issue to inform us of the problem you observed and discuss the best approach
      to fix it. If your contributions would not be captured on GitHub (social media,
      communication, educational content), you can also reach out to us on our 
      `LF Energy Slack workspace <https://slack.lfenergy.org/>`__ (#openstef channel) or attend our four-weekly 
      co-coding meetings.

    * **You are not a regular OpenSTEF user but a domain expert: you know about
      forecasting, machine learning, energy systems, time series analysis, or some
      other field where OpenSTEF could be improved.**

      Awesome — you have a focus on a specific application and domain and can
      start there. In this case, maintainers can help you figure out the best
      implementation; open an issue or pull request with a starting point, and we'll
      be happy to discuss technical approaches.

      If you prefer, you can use the `GitHub functionality for "draft" pull requests
      <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request#converting-a-pull-request-to-a-draft>`__
      and request early feedback on whatever you are working on.

    * **You are new to OpenSTEF, both as a user and contributor, and want to start
      contributing but have yet to develop a particular interest.**

      Having some previous experience with forecasting or energy systems can be very
      helpful when making open-source contributions. It helps you understand why
      things are the way they are and how they *should* be. Having first-hand
      experience and context is valuable both for what you can bring to the
      conversation and to understand where other people are coming from.

      Understanding the entire codebase is a long-term project, and nobody expects
      you to do this right away. If you are determined to get started with
      OpenSTEF and want to learn, going through the basic functionality,
      choosing something to focus on (models, feature engineering, evaluation, etc.)
      and gaining context on this area by reading the issues and pull requests
      touching these subjects is a reasonable approach.

.. _contribute_code:

Code
----

You want to implement a feature or fix a bug or help with maintenance - much
appreciated! Our library source code is found in:

* Core forecasting models: :file:`packages/openstef-models/`
* Evaluation and analysis: :file:`packages/openstef-beam/`
* Examples and tutorials: :file:`examples/`
* Tests: :file:`packages/*/tests/`

Because many people use and work on OpenSTEF, we have guidelines for keeping
our code consistent and mitigating the impact of changes.

* :doc:`code_style_guide` - Coding standards and conventions
* :doc:`development_workflow` - Pull request workflow and guidelines
* :doc:`document` - Documentation writing guidelines

Code is contributed through pull requests, so we recommend that you start at
:ref:`how-to-pull-request`. If you get stuck, please reach out on the
`LF Energy Slack workspace <https://slack.lfenergy.org/>`__ (#openstef channel) or join our four-weekly co-coding meetings.

.. _contribute_documentation:

Documentation
-------------

You, as an end-user of OpenSTEF can make a valuable contribution because you can
more clearly see the potential for improvement than a core developer. For example,
you can:

- Fix a typo
- Clarify a docstring
- Write or update an :ref:`example <examples>`
- Write or update a comprehensive :ref:`tutorial <tutorials>`
- Improve the getting started guide
- Add forecasting domain expertise to documentation

Our code is documented inline in the source code files in :file:`packages`.
Our website structure mirrors our folder structure, meaning that a narrative
document's URL roughly corresponds to its location in our folder structure:

.. grid:: 1 1 2 2

  .. grid-item:: using the library

      * :file:`examples/examples/`
      * :file:`examples/tutorials/`
      * :file:`docs/source/api/`

  .. grid-item:: information about the library

      * :file:`docs/source/user_guide/`
      * :file:`docs/source/project/`
      * :file:`docs/source/contribute/`

Instructions and guidelines for contributing documentation are found in:

* :doc:`document`
* :doc:`code_style_guide`

Documentation is contributed through pull requests, so we recommend that you start
at :ref:`how-to-pull-request`. If that feels intimidating, we encourage you to
`open an issue`_ describing what improvements you would make. If you get stuck,
please reach out on the `LF Energy Slack workspace <https://slack.lfenergy.org/>`__ 
(#openstef channel) or see our :doc:`/project/support` page for more ways to connect.

.. _other_ways_to_contribute:

Community
---------

OpenSTEF's community is built by its members! You can help by:

* Participating in our four-weekly community meetings (see :doc:`../project/support`)
* Joining discussions on the `LF Energy Slack workspace <https://slack.lfenergy.org/>`_ (#openstef channel)
* Contributing to documentation and examples

It helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!

If OpenSTEF contributes to a project that leads to a scientific publication,
please cite us following the :doc:`/project/citing` guidelines.

If you have developed an extension to OpenSTEF, please consider adding it to our
ecosystem or creating a tutorial showing how to integrate it.

.. _new_contributors:

New contributors
================

Everyone comes to the project from a different place — in terms of experience
and interest — so there is no one-size-fits-all path to getting involved. We
recommend looking at existing issue or pull request discussions, and following
the conversations during pull request reviews to get context. Or you can
deep-dive into a subset of the code-base to understand what is going on.

.. _quarterly_meetings:

Four-weekly co-coding meetings
------------------------------

Every four weeks, we host co-coding meetings to work together on OpenSTEF development,
discuss project roadmap, and support new contributors. Anyone can attend, whether
you're a seasoned contributor or just getting started. These meetings are a great
opportunity to:

* Get real-time help with your contributions
* Collaborate on complex issues
* Learn about the project architecture
* Meet other community members
* Discuss upcoming features and priorities

You can find meeting information and calendar invites on our 
`LF Energy wiki page <https://lf-energy.atlassian.net/wiki/spaces/OS/pages/32278358/Four-weekly+community+meeting>`__.
We encourage joining these meetings to get to know the people behind the GitHub handles 😉.

.. _good_first_issues:

Good first issues
-----------------

While any contributions are welcome, we have marked some issues as
particularly suited for new contributors by the label `good first issue
<https://github.com/OpenSTEF/openstef/labels/good%20first%20issue>`_. These
are well documented issues, that do not require a deep understanding of the
internals of OpenSTEF and are a great way to get started with contributing
to the project.

.. _first_contribution:

First contributions
-------------------

If this is your first open source contribution, or your first time contributing to OpenSTEF,
and you need help or guidance finding a good first issue, look no further. This section will
guide you through each step:

1. Navigate to the `issues page <https://github.com/OpenSTEF/openstef/issues/>`_.
2. Filter labels with `"good first issue" <https://github.com/OpenSTEF/openstef/labels/good%20first%20issue>`_ to find beginner-friendly tasks.
3. Click on an issue you would like to work on, and check to see if the issue has a pull request opened to resolve it.

   * A good way to judge if you chose a suitable issue is by asking yourself, "Can I independently submit a PR in 1-2 weeks?"
4. Check existing pull requests and filter by the issue number to make sure the issue is not already in progress.

   * If the issue has a pull request (is in progress), you can ask to collaborate with the existing contributor.
   * If a pull request does not exist, create a `draft pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_ and follow our pull request guidelines.
5. Please familiarize yourself with our contribution workflow and ensure you understand
   the development setup process before starting your work.

.. _get_connected:

Get connected
=============

When in doubt, we recommend going together! Get connected with our community of
active contributors, many of whom felt just like you when they started out and
are happy to welcome you and support you as you get to know how we work, and
where things are.

.. include:: _getting_help.rst

.. _managing_issues_prs:

Choose an issue
===============

In general, the OpenSTEF project does not assign issues. Issues are
"assigned" or "claimed" by opening a PR; there is no other assignment
mechanism. If you have opened such a PR, please comment on the issue thread to
avoid duplication of work. Please check if there is an existing PR for the
issue you are addressing. If there is, try to work with the author by
submitting reviews of their code or commenting on the PR rather than opening
a new PR; duplicate PRs are subject to being closed. However, if the existing
PR is stalled and the original author is unresponsive, feel free to open a new PR 
referencing the old one.

.. _how-to-pull-request:

Start a pull request
====================

The preferred way to contribute to OpenSTEF is to fork the `main
repository <https://github.com/OpenSTEF/openstef/>`__ on GitHub,
then submit a "pull request" (PR). To work on a pull request:

#. **First** set up a development environment by following the instructions in 
   :ref:`development_setup`

#. **Then** start solving the issue, following the guidance in
   :ref:`development workflow <development_workflow>`

#. **As part of verifying your changes** check that your contribution meets
   the pull request guidelines and then open a pull request.

#. **Finally** follow up with maintainers on the PR if waiting more than a few days for
   feedback. Update the pull request as needed.

If you have questions of any sort, reach out on the `LF Energy Slack workspace <https://slack.lfenergy.org/>`__ (#openstef channel) and consider
joining our :ref:`four-weekly co-coding meetings <quarterly_meetings>`.

.. _`open an issue`: https://github.com/OpenSTEF/openstef/issues/new/choose
