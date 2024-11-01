import { useCallback } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Node,
  type Edge,
  type OnConnect,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useDrop } from 'react-dnd';

import BlockNode from '@/entities/block-node/block-node';

// DropItem 타입 정의
interface DropItem {
  type: string;
}

const initialNodes: Node[] = [
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

const initialEdges: Edge[] = [{ id: 'e1-2', source: '1', target: '2' }];

const CanvasEditor = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { screenToFlowPosition } = useReactFlow();

  const onConnect: OnConnect = useCallback(
    (connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  // 노드를 중앙에 맞춰 생성하는 함수
  const addCenteredNode = useCallback(
    (clientOffset: { x: number; y: number }, item: DropItem) => {
      const initialPosition = screenToFlowPosition({
        x: clientOffset.x,
        y: clientOffset.y,
      });

      const newNode = {
        id: `${nodes.length + 1}`,
        type: 'custom',
        position: initialPosition,
        data: {
          type: item.type,
          fields: [
            { name: 'in_channels', required: true },
            { name: 'out_channels', required: true },
            { name: 'kernel_size', required: true },
          ],
        },
      };

      setNodes((nds) => [...nds, newNode]);
    },
    [nodes, screenToFlowPosition, setNodes]
  );

  // 드롭 이벤트 처리
  const [, drop] = useDrop<DropItem>(
    () => ({
      accept: 'BLOCK',
      drop: (item, monitor) => {
        const clientOffset = monitor.getClientOffset();
        if (clientOffset) {
          addCenteredNode(clientOffset, item);
        }
      },
    }),
    [addCenteredNode]
  );

  return (
    <div
      id='canvas-container'
      ref={drop}
      style={{ width: '100%', height: '100%' }}
    >
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
