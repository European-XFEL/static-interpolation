from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import Literal,Annotated

#----------------------
# Interpolation config

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
            strict = True,
            validate_assignment=True,
            validate_default=True,
            extra="forbid",
        )

class NamespacedConfigModel(StrictBaseModel):
    '''
    Enforces that for each class object FuBar in the config class
    there must be a field fu_bar.
    
    This ensures that nice things are possible like
    Exploring settings option using tab completion.

    ```config.FuBar.```
    '''
    @staticmethod
    def class_name_to_snake_case(name:str)->str:
        snake_name = name[:1].lower()
        for p, c in zip(name, name[1:]):
            if p.islower() and c.isupper():
                snake_name += "_"
            snake_name += c.lower()
        return snake_name
    
    @classmethod
    def __pydantic_on_complete__(cls) -> None:
        super().__pydantic_on_complete__()
        
        nested_models = {
            name: obj
            for name, obj in cls.__dict__.items()
            if isinstance(obj, type)
            #and issubclass(obj, BaseModel)
            and obj is not BaseModel
        }
        
        for name, model_type in nested_models.items():
            field_name = cls.class_name_to_snake_case(name)
            
            field = cls.model_fields.get(field_name,None)
            if field is None:
                raise TypeError(
                    f"{cls.__name__} class must contain a field named '{field_name}' because it contains the class '{name}'."
                )
            
class InterpolationPolicy(NamespacedConfigModel):
    """
    Configuration for Interpolation as pydantic Model.
    """
    class Method(str, Enum):
        linear = "linear"
        cubic = "cubic"
    class Boundary(str, Enum):
        clamp = "clamp"
        linear_continuation = "linear_continuation"
        reject = "reject"
    class OverlapMode(str, Enum):
        error = "error"
        first = "first"
    class Masking:
        description = "Masking options"
        
        class Strict(NamespacedConfigModel):
            kind:Literal['strict'] = 'strict'
            
        class MeanFill(NamespacedConfigModel):
            kind:Literal['mean_fill'] = 'mean_fill'
            max_masked: int = Field(
                default=3,ge=1,le=7,
                description="Maximum number of masked neighbors allowed.",
            )
            mask_nearest: bool = Field(
                default=True,
                description="Whether to mask if the nearest value is masked.",
            )
        
    method: Method = Field(
        default=Method.cubic,
        description="Interpolation types"
    )
    
    boundary: Boundary = Field(
        default=Boundary.linear_continuation,
        description="""
        Boundary treatment options.
        clamp: Sets stencil values that lie outside of the data range to the value of nearest pixel in the data.
        reject: Rejects all sampling points whose stencils contain a data point outside of the data range.
        """
    )
            
    masking: Annotated[
        Masking.Strict | Masking.MeanFill,
        Field(discriminator='kind')] = Field(
        default_factory=Masking.Strict,
        description=Masking.description
    )
    
    overlap_mode: OverlapMode = Field(
        default=OverlapMode.error,
        description="""
        What to do when an interpolation point lies on multiple data panels.
        error: Raises a ValueError
        first: Selects the first data panel on which a given interpolation point lies.
        """
    )
