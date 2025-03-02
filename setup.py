#!/usr/bin/env python
# setup.py - kept for backward compatibility

"""A setup.py file to provide backward compatibility for tools that don't support pyproject.toml."""

from setuptools import setup

if __name__ == "__main__":
    try:
        setup(name="pyhearingai")
    except:  # noqa
        print(
            "\n\nAn error occurred during installation. "
            "Please use the more modern method:\n\n"
            "    pip install .\n\n"
            "Or even better, use Poetry:\n\n"
            "    poetry install\n\n"
        )
        raise
