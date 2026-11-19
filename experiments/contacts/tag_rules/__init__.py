"""The rules that give the semantic tags of each scenario.

One file for each scenario, ``tag_rules/<scenario>.py``, with:

* ``CODEBOOK``: the tag groups. A group has a ``key``, a ``role``, a ``question``,
  the ``values`` (value -> meaning), the ``rule`` in words and the ``source`` of the
  rule (motion, contacts or program). The ``role`` is ``plan`` (what the robot does:
  the tool, the transfer type, the order, the route), ``style`` (how it does it:
  pace, heights, distances, side effects) or ``program`` (a property of the program
  text). A semantic strategy is the combination of the values of the plan groups.
* ``METHOD``: how the tag groups were found (what the reader looked at).
* ``INSIGHTS``: the main observations of the scenario, in a few sentences, for its
  page in the viewer. They are written by hand from the tag counts.
* ``label(run)``: the tags and the evidence of one ``features.Run``.

``python -m experiments.contacts.semantic <scenario>`` runs the rules and writes
``semantic/<scenario>.json``. A person can correct a rule here, or a value in the
JSON file (then set ``"checked"`` in its provenance, so that the next build keeps it).
"""
