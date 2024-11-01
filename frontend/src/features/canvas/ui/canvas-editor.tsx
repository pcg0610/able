import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type OnConnect,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useNodeDropHandler } from '@features/canvas/model/useNodeDropHandler.model';
import { initialNodes, initialEdges } from '@features/canvas/model/initialData';

import BlockNode from '@/entities/block-node/block-node';

const CanvasEditor = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { screenToFlowPosition } = useReactFlow();

  const onConnect: OnConnect = (connection) =>
    setEdges((eds) => addEdge(connection, eds));

  const { dropRef } = useNodeDropHandler({ setNodes, screenToFlowPosition });

  return (
    <div ref={dropRef} style={{ width: '100%', height: '100%' }}>
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
