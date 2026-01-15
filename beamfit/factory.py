registered_objects = {}


def register(objtype: str, name: str, createfun: callable):
    """
    Registers the creation function with the given name and an object type

    Parameters
    ----------
    objtype : str
        The type of the object to register
    name : str
        The name of the object to register
    createfun : function
        The constructor of the object
    """
    if objtype not in registered_objects:
        registered_objects[objtype] = {}
    if name in registered_objects[objtype]:
        raise ValueError(f"Name '{name}' of type '{objtype}' is already registered")
    registered_objects[objtype][name] = createfun


def create(objtype: str, name: str, **kwargs):
    """
    Returns an object created with the previously registered constructor. Any additional arguments
    are passed to the constructor as keyword arguments.

    objtype : str
        The type of object to create
    name : str
        The name of the object to create
    """
    return registered_objects[objtype][name](**kwargs)


def unregister(objtype: str, name: str):
    """
    Remove an object from the registry.

    Parameters
    ----------
    objtype : str
        The type of the object to unregister
    name : str
        The name of the object to unregister
    """
    registered_objects[objtype].pop(name)


def get_names(objtype: str) -> list[str]:
    """
    Return a list of names registered to this object type

    Parameters
    ----------
    objtype : str
        The type of the object to look up

    Returns
    -------
    list[str]
        The names of the registered objects
    """
    return list(registered_objects[objtype].keys())
