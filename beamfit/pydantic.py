import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class NumpyArray(np.ndarray):
    """NumPy array subclass with Pydantic support"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate_array(v):
            return np.asarray(v, dtype=float)

        return core_schema.no_info_after_validator_function(
            validate_array,
            core_schema.union_schema(
                [
                    core_schema.list_schema(core_schema.float_schema()),
                    core_schema.is_instance_schema(np.ndarray),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.tolist()
            ),
        )
