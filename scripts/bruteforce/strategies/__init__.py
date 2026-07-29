"""Strategy modules. Every module here is imported by ``registry.load_all()``.

One module per family. A module's only job is to declare ``@register``-ed build
functions; it must not run a sweep or touch the filesystem at import time,
because ``load_all`` imports all of them just to enumerate the registry.
"""
