from collections import defaultdict, deque
from typing import Dict, Tuple
from src.canvas.schemas import Canvas, CanvasBlock

def check_cycle(canvas: Canvas) -> Tuple[bool, list[CanvasBlock]]:
    """
    Canvas 객체 내의 방향 그래프에서 사이클이 존재하는지 위상 정렬을 통해 판별하고,
    사이클에 포함된 블록들을 반환

    :param canvas: Canvas 객체, 블록과 엣지 정보를 포함
    :return: (사이클 존재 여부, 사이클에 포함된 블록 리스트)
    """
    adj_blocks: dict[str, list[str]] = defaultdict(list)
    in_degree = defaultdict(int)

    for edge in canvas.edges:
        adj_blocks[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue = deque([block.id for block in canvas.blocks if in_degree[block.id] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for adj_block in adj_blocks.pop(node, []):
            in_degree[adj_block] -= 1
            if in_degree[adj_block] == 0:
                queue.append(adj_block)

    # 모든 노드를 방문하지 못했다면 사이클이 존재
    if visited_count != len(canvas.blocks):
        cycle_blocks = [
            block for block in canvas.blocks if in_degree[block.id] > 0
        ]
        return True, cycle_blocks

    return False, []