"""
spore.dashboard — Streamlit dashboard package.

Sub-packages
------------
pages/
    One module per dashboard tab.  Each exposes ``render_tab()`` (or
    ``render_prompt_trajectory_viewer()`` for the Trajectory tab).

components/
    Reusable Streamlit widgets and plot builders shared across pages.

data/
    Cached data-loading and computation helpers.

Both the old import path (``spore.app.*``) and this new path remain valid.
"""
