import { useState, useCallback } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  addEdge,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import BlockNode from '@/entities/canvas/block-node';

const initialNodes = [
  {
    id: '1',
    type: 'custom', // 노드 유형을 custom으로 설정
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

const initialEdges = [{ id: 'e1-2', source: '1', target: '2' }];

const CanvasEditor = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    []
  );

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={{ custom: BlockNode }}
      >
        <Controls position='bottom-center' orientation='horizontal' />
        <Background variant={BackgroundVariant.Dots} />
      </ReactFlow>
    </div>
  );
};

export default CanvasEditor;
