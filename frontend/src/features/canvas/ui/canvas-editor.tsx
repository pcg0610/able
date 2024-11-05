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
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { Canvas } from '@features/canvas/ui/canvas-editor.style';
import { useNodeDropHandler } from '@features/canvas/model/use-node-drop-handler.model';
import { initialNodes, initialEdges } from '@features/canvas/model/initialData';
import { useCanvas } from '@features/canvas/api/use-canvas';

import BlockNode from '@entities/block-node/block-node';

const CanvasEditor = () => {
  const { data } = useCanvas('춘식이');

  const [nodes, setNodes, onNodesChange] = useNodesState(
    data ? data.nodes : initialNodes
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    data ? data.edges : initialEdges
  );
  const { screenToFlowPosition } = useReactFlow();

  const onConnect: OnConnect = (connection) =>
    setEdges((eds) =>
      addEdge(
        {
          ...connection,
          type: 'smoothstep',
          markerEnd: { type: MarkerType.ArrowClosed, width: 30, height: 30 },
        },
        eds
      )
    );

  const { dropRef } = useNodeDropHandler({ setNodes, screenToFlowPosition });

  return (
    <Canvas ref={dropRef}>
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
    </Canvas>
  );
};

export default CanvasEditor;
