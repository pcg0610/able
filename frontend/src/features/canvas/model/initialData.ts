import { Node, Edge } from '@xyflow/react';

// 초기 노드 데이터
export const initialNodes: Node[] = [
  {
    id: '1',
    type: 'custom',
    position: { x: 0, y: 0 },
    data: {
      type: 'layer',
      fields: [
        { name: 'in_channels', required: true },
        { name: 'out_channels', required: true },
        { name: 'kernel_size', required: true },
      ],
      onFieldChange: (fieldName: string, value: string) => {
        console.log(`Field ${fieldName} updated with value: ${value}`);
      },
    },
  },
  {
    id: '2',
    type: 'custom',
    position: { x: 250, y: 0 },
    data: {
      type: 'activation',
      fields: [{ name: 'activation_type', required: true }],
      onFieldChange: (fieldName: string, value: string) => {
        console.log(`Field ${fieldName} updated with value: ${value}`);
      },
    },
  },
];

// 초기 엣지 데이터
export const initialEdges: Edge[] = [{ id: 'e1-2', source: '1', target: '2' }];
