import {
  Node as XYFlowNode,
  Edge as XYFlowEdge,
  MarkerType,
} from '@xyflow/react';

import {
  BlockSchema,
  CanvasResponse,
  EdgeSchema,
} from '@features/canvas/types/canvas.type';

export const transformCanvasResponse = (response: CanvasResponse) => {
  const transformedNodes: XYFlowNode[] = response.data.canvas.blocks.map(
    (block) => ({
      id: block.id,
      type: 'custom',
      position: JSON.parse(block.position),
      data: {
        block: {
          type: block.type,
          name: block.name,
          fields: block.args,
        },
      },
    })
  );

  const transformedEdges: XYFlowEdge[] = response.data.canvas.edges.map(
    (edge) => ({
      id: edge.id,
      source: edge.source.toString(),
      target: edge.target.toString(),
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed, width: 30, height: 30 },
    })
  );

  return { nodes: transformedNodes, edges: transformedEdges };
};

export const transformNodesToBlocks = (nodes: XYFlowNode[]): BlockSchema[] => {
  return nodes.map((node) => ({
    id: node.id,
    name: (node.data as any).block.name,
    type: (node.data as any).block.type,
    args: (node.data as any).block.fields,
    position: JSON.stringify(node.position),
  }));
};

export const transformEdgesToEdgeResponse = (
  edges: XYFlowEdge[]
): EdgeSchema[] => {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
  }));
};
