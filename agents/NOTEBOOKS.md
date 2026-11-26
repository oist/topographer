# NOTEBOOKS.md

Instructions for working with notebooks.

- All notebooks are stored in `notebooks/`.
- Notebooks under `notebooks/archive/` are unsupported historical artifacts and
  are excluded from maintained documentation and verification.
- Marimo notebooks are Python `.py` files.
- Notebook outputs and session artifacts are stored in `notebooks/__marimo__/`.
- Check notebook syntax with `uv run --extra docs marimo check`
- Run a notebook with `uv run --extra docs python`


## Marimo
<!-- Source: https://docs.marimo.io/guides/best_practices/ -->

- **Use global variables sparingly.** Keep the number of global variables in your program small to avoid name collisions. If you have intermediate variables, encapsulate them in functions or prefix them with an underscore (`_tmp = ...`) to make them local to a cell.

- **Use descriptive names.** Use descriptive variable names, especially for global variables. This will help you minimize name clashes, and will also result in better code.

- **Use functions.** Encapsulate logic into functions to avoid polluting the global namespace with temporary or intermediate variables, and to avoid code duplication.

- **Use Python modules.** If your notebook gets too long, split complex logic into helper Python modules and import them into your notebook.

- **Minimize mutations.** marimo does not track mutations to objects. Try to
only mutate an object in the cell that creates it, or create new objects
instead of mutating existing ones.

- **Don't use state and `on_change` handlers.** Don't use `on_change` handlers to react to UI interactions. Instead, use marimo's built-in [reactive execution for interactive elements](https://docs.marimo.io/guides/interactivity/).

- **Write idempotent cells.** Write cells whose outputs and behavior are the same when given the same inputs (references); such cells are called idempotent. This will help you avoid bugs and cache expensive intermediate computations.

## Altair

- Altair has a known issue when sorting ordinal axis labels by an attribute other than lexicographic label order.

## Naming convention
<!-- Source: https://cookiecutter-data-science.drivendata.org/using-the-template/#open-a-notebook -->

A scheme that looks like this:

`0-01_data-source.py`
`0-01` - Helps keep work in chronological order. The structure is `PHASE-NOTEBOOK_NUMBER`. `NOTEBOOK_NUMBER` is just the n-th notebook in that phase to be created. For phases of the project, we generally use a scheme like the following, but you are welcome to design your own conventions:
    `0` - Data exploration - often just for exploratory work
    `1` - Data cleaning and feature creation - often writes data to data/processed or data/interim
    `2` - Visualizations - often writes publication-ready viz to reports
    `3` - Modeling - training machine learning models
    `4` - Publication - Notebooks that get turned directly into reports
