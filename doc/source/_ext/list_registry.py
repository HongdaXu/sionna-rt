#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sphinx extension: custom RST roles for Sionna documentation.

Provides the ``list-registry`` role, which imports a named registry object and
renders its entries as a formatted inline list, e.g.
``("entry1" | "entry2" | "entry3")``.
"""

import importlib
from docutils import nodes

def _list_registry_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    try:
        module_name, registry_name = text.rsplit(".", 1)
        module = importlib.import_module(module_name)
        registry = getattr(module, registry_name)
        output_list = registry.list()
        formatted = "(" + " | ".join(f'"{entry}"' for entry in output_list) + ")"
        node = nodes.inline(rawtext, formatted, **options)
        return [node], []
    except Exception as e:
        error = inliner.reporter.error(
            f"Error processing role 'list-registry': {e}",
            nodes.literal_block(rawtext, rawtext),
            line=lineno,
        )
        return [inliner.problematic(rawtext, rawtext, error)], [error]


def setup(app):
    app.add_role("list-registry", _list_registry_role)
    return {
        "version": "0.1", 
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
