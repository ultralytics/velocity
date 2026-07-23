# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""String formatting utilities."""


def filenamesplit(string):
    """Split a full filename string into path, file, and extension; returns a tuple (path, file, extension, fileext)."""
    i = string.rfind("/") + 1
    j = string.rfind(".")
    path, file, extension = string[:i], string[i:j], string[j:]
    return path, file, extension, file + extension


def printd(dictionary):  # print dictionary
    """Print each key-value pair in a dictionary, with keys aligned for readability."""
    for tag in dictionary:
        print(f"{tag!s:>40}: {dictionary[tag]}")
