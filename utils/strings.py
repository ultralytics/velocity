# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license


def filenamesplit(string):  # splits a full filename string into path, file, and extension.
    """Splits a full filename string into path, file, and extension; returns a tuple (path, file, extension,
    fileext).
    """
    i = string.rfind("/") + 1
    j = string.rfind(".")
    path, file, extension = string[:i], string[i:j], string[j:]
    return path, file, extension, file + extension


def printd(dictionary):  # print dictionary
    """Prints each key-value pair in a dictionary, with keys aligned for readability."""
    for tag in dictionary.keys():
        print("%40s: %s" % (tag, dictionary[tag]))
