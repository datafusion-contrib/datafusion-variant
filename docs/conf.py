# SPDX-License-Identifier: Apache-2.0

project = "DataFusion Variant"
extensions = ["myst_parser"]
# README.md is the GitHub landing page for docs/ and is included by index.md.
exclude_patterns = ["README.md"]
templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_center": [],
    "secondary_sidebar_items": [],
    "footer_start": [],
    "footer_end": [],
    "pygments_light_style": "vs",
    "pygments_dark_style": "github-dark",
}
html_static_path = ["_static"]
html_css_files = ["theme_overrides.css"]
html_sidebars = {"**": ["docs-sidebar.html"]}

# Render `--` and quotes as typed, as GitHub does.
smartquotes = False
myst_heading_anchors = 3
myst_enable_extensions = ["alert"]
# Pygments' SQL lexer rejects JSON literals such as `{`.
suppress_warnings = ["misc.highlighting_failure"]
