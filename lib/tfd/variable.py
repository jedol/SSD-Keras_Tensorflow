import tensorflow as tf


def filter_variables(collection, includes=[], excludes=[]):
    if not isinstance(includes, list) and not isinstance(includes, tuple):
        includes = [includes]
    if not isinstance(excludes, list) and not isinstance(excludes, tuple):
        excludes = [excludes]

    include_vars = list()
    if len(includes):
        include_vars.extend([v for v in collection for i in includes if i in v.name])
    else:
        include_vars.extend(collection)

    exclude_vars = list()
    if len(excludes):
        exclude_vars.extend([v for v in include_vars for e in excludes if e not in v.name])
    else:
        exclude_vars.extend(include_vars)

    return exclude_vars


def global_variables(scopes=[], includes=[], excludes=[]):
    variables = list()
    if len(scopes):
        for scope in scopes:
            with tf.variable_scope(scope):
                collection = tf.get_variable_scope().global_variables()
                variables.extend(filter_variables(collection, includes, excludes))
    else:
        collection = tf.global_variables()
        variables.extend(filter_variables(collection, includes, excludes))

    return variables


def trainable_variables(scopes=[], includes=[], excludes=[]):
    variables = list()
    if len(scopes):
        for scope in scopes:
            with tf.variable_scope(scope):
                collection = tf.get_variable_scope().trainable_variables()
                variables.extend(filter_variables(collection, includes, excludes))
    else:
        collection = tf.trainable_variables()
        variables.extend(filter_variables(collection, includes, excludes))

    return variables