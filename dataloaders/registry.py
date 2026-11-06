"""
Registry for dataset classes to avoid circular imports.
"""


class DatasetRegistry:
    """Registry for dataset classes to avoid circular imports."""

    _registry = {}

    @classmethod
    def register(cls, name, dataset_class):
        """Register a dataset class with a name."""
        cls._registry[name] = dataset_class
        return dataset_class

    @classmethod
    def get(cls, name):
        """Get a dataset class by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all(cls):
        """Get all registered dataset classes."""
        return cls._registry
