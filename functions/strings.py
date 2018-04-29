def filenamesplit(string):  # splits a full filename string into path, file, and extension.
    # Example:  path, file, extension, fileext = filenamesplit('/Users/glennjocher/Downloads/IMG_4124.JPG')
    i = string.rfind('/') + 1
    j = string.rfind('.')
    path, file, extension = string[:i], string[i:j], string[j:]
    return path, file, extension, file + extension


def printd(dictionary):  # print dictionary
    for tag in dictionary.keys():
        print('%40s: %s' % (tag, dictionary[tag]))
