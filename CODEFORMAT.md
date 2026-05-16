Formatting — vertical alignment

For dataclasses, @dataclass, TypedDict-style field lists, and config structs with
multiple annotated attributes, format like:

    field_name_a : str   = "default"
    field_name_b : int   = 0
    field_name_c : float = 1.0

Rules:
- Align `:` in a column after the longest field name in the block.
- Align `=` in a column after the longest type annotation in the block.
- One space before and after `:`; one space before and after `=` unless Black would
  forbid it — prefer readability over strict PEP 8 line length for these blocks.
- Apply the same style to grouped module-level constants or kwargs dicts only when
  I ask or the block is clearly a “schema” (3+ related keys).
- Do NOT use this for normal functions, random assignments, or imports.
- Black/ruff may undo this; if the project uses a formatter that forbids alignment,
  mention that and align only in files we do not auto-format.