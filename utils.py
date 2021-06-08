def complement_feature(feature):
    """
    given a feature, return the complement
    of that feature.

    e.g. 
    input feature is "Y" (young) output is
    the complement, "A" (adult)
    """

    features_set = [["Y", "A"],["I", "D"]]
    for f_set in features_set:
        if feature in f_set:
            f_set.remove(feature)
            return f_set[0]
    return 0

