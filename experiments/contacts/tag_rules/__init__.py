"""The rules that give the semantic tags of each scenario.

One file for each scenario, ``tag_rules/<scenario>.py``, with:

* ``CODEBOOK``: the tag groups. A group has a ``key``, a ``question``, the
  ``values`` (value -> meaning), the ``rule`` in words and the ``source`` of the
  rule (motion, contacts or program).
* ``METHOD``: how the tag groups were found (what the reader looked at).
* ``INSIGHTS``: the main observations of the scenario, in a few sentences, for its
  page in the viewer. They are written by hand from the tag counts.
* ``label(run)``: the tags and the evidence of one ``features.Run``.

``python -m experiments.contacts.semantic <scenario>`` runs the rules and writes
``semantic/<scenario>.json``. A person can correct a rule here, or a value in the
JSON file (then set ``"checked"`` in its provenance, so that the next build keeps it).
"""
