
import torch
import json
import base64
import io
import numpy as np

from PIL import Image
from fastapi import APIRouter, Body
from src.analysis.utils import read_blocks
from src.deploy.schemas import InferenceResponse
from src.file.path_manager import PathManager
from src.file.utils import get_file
from src.response.utils import ok
from src.train.schemas import TrainResultMetadata
from src.train.utils import create_data_preprocessor, split_blocks
from src.utils import str_to_json

router = APIRouter()
path_manager = PathManager()

@router.post("/model")
async def path_name_route(image: str = Body(...)):
    
    project_name = "string"
    train_result = "20241108_163451"
    checkpoint = "final"
    
    train_result_metadata_path = path_manager.get_train_result_path(project_name, train_result) / "metadata.json"
    metadata = TrainResultMetadata(**str_to_json(get_file(train_result_metadata_path)))

    #block_graph.json 파일에서 블록 읽어오기
    block_graph_path = path_manager.get_train_result_path(project_name, train_result) / "block_graph.json"
    block_graph = read_blocks(block_graph_path)

    # base64를 이미지로 변환 
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    
    # 블록 카테고리 별로 나누기
    _, transform_blocks, _, _, _ = split_blocks(block_graph.blocks)
    transforms = create_data_preprocessor(transform_blocks)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    image: torch.Tensor = transforms(image)
    image = image.unsqueeze(0).to(device=device)

    model = torch.load(path_manager.get_checkpoint_path(project_name, train_result, checkpoint) / "model.pth")

    model.to(device)

    model.eval()
    predicted = model(image)

    top_values, top_indices = predicted.topk(1, dim=1)
    top_values = top_values[0].cpu().detach().numpy()
    top_indices = top_indices[0].cpu().detach().numpy()
        
    predicted_label = metadata.classes[top_indices[0]]
    max_value = top_values[0]
    
    return ok(
        data=InferenceResponse(
            label = predicted_label,
            probability = max_value
        )
    )
