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

import * as S from '@features/canvas/ui/canvas-editor.style';
import Common from '@shared/styles/common';
import { useNodeDropHandler } from '@features/canvas/model/use-node-drop-handler.model';
import { initialNodes, initialEdges } from '@features/canvas/model/initialData';
import { useCanvas } from '@/features/canvas/api/use-canvas.query';

import BlockNode from '@entities/block-node/block-node';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import SaveIcon from '@icons/save.svg?react';

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

  const handleTrainButtonClick = () => {
    console.log('실행');
  };

  const handleSavaButtonClick = () => {
    console.log('저장');
    console.log(nodes);
    console.log(edges);
  };

  return (
    <S.Canvas ref={dropRef}>
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
      <S.OverlayButton>
        <BasicButton
          text='실행'
          icon={<PlayIcon width={13} height={16} />}
          width='5.5rem'
          onClick={handleTrainButtonClick}
        />
        <BasicButton
          text='저장'
          color={Common.colors.primary}
          backgroundColor={Common.colors.secondary}
          icon={<SaveIcon />}
          width='5.5rem'
          onClick={handleSavaButtonClick}
        />
      </S.OverlayButton>
    </S.Canvas>
  );
};

export default CanvasEditor;
