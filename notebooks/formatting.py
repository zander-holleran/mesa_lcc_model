import pandas as pd
from IPython import get_ipython

_DOLLAR_COLS = set()
_DOLLAR_TOKENS = set()
_ELAPSED_TIME_COLS = set()
_ELAPSED_TIME_TOKENS = set()


def _matches(col, exact_names, tokens):
    col_str = str(col)
    col_lower = col_str.lower()
    return col in exact_names or any(token in col_lower for token in tokens)


def _is_missing_scalar(value):
    return pd.api.types.is_scalar(value) and pd.isna(value)


def _format_three_decimals(value):
    if _is_missing_scalar(value):
        return "—"

    try:
        return f"{float(value):,.3f}"
    except (TypeError, ValueError):
        return value


def _format_dollar(value):
    if _is_missing_scalar(value):
        return "—"

    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return _format_three_decimals(value)


def _format_elapsed_time(value):
    if _is_missing_scalar(value):
        return "—"

    try:
        total_seconds = int(round(float(value) * 60))
    except (TypeError, ValueError):
        return _format_three_decimals(value)

    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{sign}{minutes:02d}:{seconds:02d}"


def _df_html_formatter(df):
    fmt = {}

    for col in df.columns:
        if _matches(col, _DOLLAR_COLS, _DOLLAR_TOKENS):
            fmt[col] = _format_dollar
        elif _matches(col, _ELAPSED_TIME_COLS, _ELAPSED_TIME_TOKENS):
            fmt[col] = _format_elapsed_time
        elif (
            pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_integer_dtype(df[col])
        ):
            fmt[col] = _format_three_decimals

    if fmt:
        return df.style.format(fmt, na_rep="—")._repr_html_()
    return df._repr_html_()


def _install_dataframe_formatter():
    ip = get_ipython()
    if ip:
        html_fmt = ip.display_formatter.formatters["text/html"]
        html_fmt.for_type(pd.DataFrame, _df_html_formatter)


def register_dollar_cols(cols=None, tokens=None):
    """Register dollar-formatted columns by exact names and/or token matches.

    Call once in notebook setup. Any DataFrame displayed afterward will
    auto-format matching columns. Underlying data stays numeric.

    Parameters
    ----------
    cols : list[str] | None
        Exact column names to format as dollars.
    tokens : list[str] | None
        Substrings; any column containing one will be formatted as dollars.
    """
    if cols:
        _DOLLAR_COLS.update(cols)
    if tokens:
        _DOLLAR_TOKENS.update(token.lower() for token in tokens)
    _install_dataframe_formatter()


def register_elapsed_time(cols=None, tokens=None):
    """Register elapsed-time columns to display as MM:SS.

    Call once in notebook setup. Any DataFrame displayed afterward will
    auto-format matching columns. Underlying data stays numeric.

    Parameters
    ----------
    cols : list[str] | None
        Exact column names to format as elapsed time.
    tokens : list[str] | None
        Substrings; any column containing one will be formatted as elapsed time.

    Notes
    -----
    Values are assumed to be elapsed minutes and may be floats.
    """
    if cols:
        _ELAPSED_TIME_COLS.update(cols)
    if tokens:
        _ELAPSED_TIME_TOKENS.update(token.lower() for token in tokens)
    _install_dataframe_formatter()
