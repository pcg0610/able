import logging
import inspect
import importlib
from typing import Iterator, Union, Any, Dict, Tuple, List, Callable
from torch import nn, optim
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from torchvision import transforms
from src.block.enums import BlockType
from src.block.schemas import Block

logger = logging.getLogger(__name__)

MODULE_MAP = {
    BlockType.TRANSFORM: "torchvision.transforms",
    BlockType.LAYER: "torch.nn",
    BlockType.ACTIVATION: "torch.nn",
    BlockType.LOSS: "torch.nn",
    BlockType.OPERATION: "torch.nn.functional",
    BlockType.OPTIMIZER: "torch.optim",
    BlockType.MODULE: "torch.nn",
    BlockType.DATA: "torch.utils.data",
    BlockType.INTERPRETER: "torch.fx"
}

def dynamic_class_loader(module_path: str, class_name: str):
    """ 동적으로 모듈에서 클래스를 로드합니다. """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}. Error: {e}")
        raise
    except AttributeError:
        logger.error(f"{class_name} not found in module {module_path}")
        raise ValueError(f"{class_name} not found in module {module_path}")

def convert_block_to_module(block: Block, parameters: Iterator[nn.Parameter] = None) -> Union[
    nn.Module,
    optim.Optimizer,
    transforms.Compose,
    transforms.Resize,
    DataLoader,
    GraphModule,
    Callable[..., Any]
]:
    """ 주어진 Block 객체를 PyTorch 모듈, 옵티마이저, 또는 함수로 변환합니다. """
    module_path = MODULE_MAP.get(block.type)
    if not module_path:
        raise ValueError(f"Unsupported block type: {block.type.value}")

    cls = dynamic_class_loader(module_path, block.name)

    # 유효한 파라미터만 사용하도록 필터링
    valid_args, ignored_args, missing_args = validate_params(cls, block.args)

    if ignored_args:
        logger.warning(f"Ignored invalid arguments for {block.name}: {ignored_args}")

    if missing_args:
        raise ValueError(f"Missing required arguments for {block.name}: {missing_args}")

    # 각 블록 타입에 따른 객체 생성
    try:
        if block.type == BlockType.OPTIMIZER:
            if parameters is None:
                raise ValueError("Parameters must be provided for optimizer initialization.")
            logger.debug(f"Initializing optimizer {block.name} with args {valid_args}")
            return cls(parameters, **valid_args)

        elif block.type in {BlockType.LAYER, BlockType.ACTIVATION, BlockType.LOSS, BlockType.MODULE}:
            logger.debug(f"Initializing module {block.name} with args {valid_args}")
            return cls(**valid_args)

        elif block.type == BlockType.OPERATION:
            logger.debug(f"Initializing operation {block.name} with args {valid_args}")
            return lambda *args, **kwargs: cls(*args, **kwargs, **valid_args)

        elif block.type == BlockType.TRANSFORM:
            # 연속적 변환 지원
            logger.debug(f"Initializing transform {block.name} with args {valid_args}")
            if block.name == "Compose" and "transforms" in block.args:
                transforms_list = [
                    dynamic_class_loader("torchvision.transforms", t['name'])(**t.get('args', {}))
                    for t in block.args['transforms']
                ]
                return cls(transforms_list)
            return cls(**valid_args)

        elif block.type == BlockType.DATA:
            logger.debug(f"Initializing data module {block.name} with args {valid_args}")
            return cls(**valid_args)

        elif block.type == BlockType.INTERPRETER:
            logger.debug(f"Initializing interpreter {block.name} with args {valid_args}")
            return cls(**valid_args)

        else:
            raise ValueError(f"Block type '{block.type.value}' is not supported.")

    except TypeError as e:
        logger.error(f"Failed to initialize {block.name} with args {valid_args}. Error: {e}")
        raise

def validate_params(cls, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    동적으로 로드된 클래스의 파라미터가 유효한지 확인하고,
    유효한 파라미터와 무시된 파라미터 목록을 반환합니다.
    """
    signature = inspect.signature(cls)
    valid_args = {}
    ignored_args = []
    missing_args = []

    for key, value in args.items():
        if key in signature.parameters:
            valid_args[key] = value
        else:
            ignored_args.append(key)

    # 기본값을 사용해야 하는 필수 파라미터를 찾기
    for param_name, param in signature.parameters.items():
        if param_name not in valid_args and param.default is param.empty:
            missing_args.append(param_name)

    return valid_args, ignored_args, missing_args
