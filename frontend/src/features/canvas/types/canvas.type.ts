import { Response } from '@/shared/types/response.type';
import type { Block } from '@features/canvas/types/block.type';
import { Edge, Node } from '@xyflow/react';

export interface EdgeResponse {
  id: string;
  source: string;
  target: string;
}

export interface BlockResponse extends Block {
  id: string;
  position: string;
}

export interface CanvasResponse extends Response {
  data: { canvas: { blocks: BlockResponse[]; edges: EdgeResponse[] } };
}

export interface TransformedCanvas {
  nodes: Node[];
  edges: Edge[];
}
