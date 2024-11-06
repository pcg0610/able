import { type Node, type Edge } from '@xyflow/react';

import d3Hierarchy from '@features/train/utils/d3-hierarchy.utils';

export type Direction = 'TB' | 'LR' | 'RL' | 'BT';

export type LayoutAlgorithmOptions = {
  direction: Direction;
  spacing: [number, number];
};

export type LayoutAlgorithm = (
  nodes: Node[],
  edges: Edge[],
  options: LayoutAlgorithmOptions
) => Promise<{ nodes: Node[]; edges: Edge[] }>;

export default {
  'd3-hierarchy': d3Hierarchy,
};
