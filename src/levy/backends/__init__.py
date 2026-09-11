#    Copyright (C) 2017 José M. Miotto
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""Which array library evaluates the distribution.

NumPy is the default and always will be. The optional ``torch`` backend exists
for one reason: it makes the log likelihood differentiable, so a stable
distribution can sit inside a larger model and be fitted by gradient descent
along with everything else. It also runs on a GPU, which matters only for very
large samples.

Selection, in order of precedence:

1. an explicit ``backend='torch'`` argument;
2. whatever :func:`set_backend` or :func:`using` last selected;
3. automatic: if any argument is a ``torch.Tensor``, torch; otherwise NumPy.

Rule 3 is what makes the common case work without ceremony -- hand the
functions tensors and you get tensors back, with gradients attached. It costs
nothing when torch is absent, because the check is a lookup in ``sys.modules``
and not an import attempt.

Examples
--------
>>> get().name
'numpy'
>>> with using('numpy'):
...     get().name
'numpy'
"""

from contextlib import contextmanager

from levy._compat import loaded, require

__all__ = ['get', 'set_backend', 'using']

_BACKENDS = ('numpy', 'torch')

#: What set_backend last selected, or None for automatic.
_selected = None


def _load(name):
    """Import and return a backend module.

    Parameters
    ----------
    name : {'numpy', 'torch'}
        Which backend.

    Returns
    -------
    module
        The backend module, which exposes ``name``, ``pdf``, ``cdf`` and
        ``neglog``.

    Raises
    ------
    ValueError
        If `name` is not a known backend.
    ImportError
        If the backend's library is not installed, naming the extra.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f'unknown backend {name!r}; available: {", ".join(_BACKENDS)}')
    if name == 'torch':
        require('torch')
        from levy.backends import _torch

        return _torch
    from levy.backends import _numpy

    return _numpy


def _is_tensor(value):
    """Report whether `value` is a torch tensor, without importing torch.

    Parameters
    ----------
    value : object
        Anything.

    Returns
    -------
    bool
        True only if torch is already imported and `value` is one of its
        tensors. If nobody has imported torch, nothing can be a tensor.
    """
    torch = loaded('torch')
    return torch is not None and isinstance(value, torch.Tensor)


def get(name=None, *values):
    """Return the backend to use for a call.

    Parameters
    ----------
    name : {'numpy', 'torch'}, optional
        An explicit choice, which wins over everything else.
    *values
        The arguments of the call. If any is a ``torch.Tensor`` and no explicit
        or global choice is in force, the torch backend is selected.

    Returns
    -------
    module
        The backend module.

    Examples
    --------
    >>> get().name
    'numpy'
    >>> get('numpy').name
    'numpy'
    """
    if name is not None:
        return _load(name)
    if _selected is not None:
        return _load(_selected)
    if any(_is_tensor(value) for value in values):
        return _load('torch')
    return _load('numpy')


def set_backend(name):
    """Select a backend for every subsequent call.

    Parameters
    ----------
    name : {'numpy', 'torch'} or None
        The backend, or None to go back to automatic selection.

    Returns
    -------
    str or None
        Whatever was selected before, so a caller can restore it.

    Raises
    ------
    ValueError
        If `name` is not a known backend.
    ImportError
        If the backend's library is not installed.
    """
    global _selected
    previous = _selected
    if name is not None:
        _load(name)          # fail here, not at the first evaluation
    _selected = name
    return previous


@contextmanager
def using(name):
    """Select a backend for the duration of a block.

    Parameters
    ----------
    name : {'numpy', 'torch'} or None
        The backend, or None for automatic selection.

    Yields
    ------
    module
        The backend in force inside the block.
    """
    previous = set_backend(name)
    try:
        yield _load(name) if name is not None else get()
    finally:
        set_backend(previous)
