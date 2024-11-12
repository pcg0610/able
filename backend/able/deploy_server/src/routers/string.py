import torch
import base64
import io
import numpy as np

from PIL import Image
from fastapi import APIRouter, Body

from deploy_server.src.schemas import InferenceResponse
from src.file.path_manager import PathManager
from src.file.utils import get_file
from src.response.utils import ok
from src.train.schemas import TrainResultMetadata
from src.train.utils import create_data_preprocessor, split_blocks
from src.utils import str_to_json
from src.analysis.utils import read_blocks

router = APIRouter()
path_manager = PathManager()

@router.post("/string")
async def string_route(image: str = Body(...)):
    project_name, train_result, checkpoint = "string", "string", "string"
    metadata_path = path_manager.get_train_result_path(project_name, train_result) / "metadata.json"
    metadata = TrainResultMetadata(**str_to_json(get_file(metadata_path)))

    # Load block graph
    block_graph_path = path_manager.get_train_result_path(project_name, train_result) / "block_graph.json"
    block_graph = read_blocks(block_graph_path)

    # Decode and preprocess image
    image = np.array(Image.open(io.BytesIO(base64.b64decode(image))))
    _, transform_blocks, _, _, _ = split_blocks(block_graph.blocks)
    transforms = create_data_preprocessor(transform_blocks)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image_tensor = transforms(image).unsqueeze(0).to(device)

    # Load model and make prediction
    model = torch.load(path_manager.get_checkpoint_path(project_name, train_result, checkpoint) / "model.pth")
    model.to(device).eval()
    predicted = model(image_tensor)
    top_values, top_indices = predicted.topk(1, dim=1)

    predicted_label = metadata.classes[top_indices[0].item()]
    max_value = top_values[0].item()

    return ok(data=InferenceResponse(label=predicted_label, probability=max_value))
    