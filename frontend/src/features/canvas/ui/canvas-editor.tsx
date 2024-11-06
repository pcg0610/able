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
import { useCallback } from 'react';

import * as S from '@features/canvas/ui/canvas-editor.style';
import Common from '@shared/styles/common';
import { useNodeDropHandler } from '@features/canvas/model/use-node-drop-handler.model';
import {
  initialNodes,
  initialEdges,
} from '@/features/canvas/model/initial-data';
import { useFetchCanvas } from '@/features/canvas/api/use-canvas.query';
import { useSaveCanvas } from '@features/canvas/api/use-canvas.mutation';
import type { BlockItem } from '@features/canvas/types/block.type';
import {
  transformEdgesToEdgeResponse,
  transformNodesToBlocks,
} from '@features/canvas/utils/canvas-transformer.util';

import BlockNode from '@entities/block-node/block-node';
import BasicButton from '@shared/ui/button/basic-button';
import PlayIcon from '@icons/play.svg?react';
import SaveIcon from '@icons/save.svg?react';

const CanvasEditor = () => {
  const { data } = useFetchCanvas('춘식이');
  const { mutate: saveCanvas } = useSaveCanvas();

  const [nodes, setNodes, onNodesChange] = useNodesState(
    data ? data.nodes : initialNodes
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    data ? data.edges : initialEdges
  );
  const { screenToFlowPosition } = useReactFlow();
  const { dropRef } = useNodeDropHandler({ setNodes, screenToFlowPosition });

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

  const handleTrainButtonClick = () => {
    console.log('실행');
  };

  const handleSavaButtonClick = () => {
    console.log('저장');
    const transformedBlocks = transformNodesToBlocks(nodes);
    const transformedEdges = transformEdgesToEdgeResponse(edges);

    saveCanvas({
      projectName: '춘식이',
      canvas: { blocks: transformedBlocks, edges: transformedEdges },
    });
  };

  const handleFieldChange = useCallback(
    (nodeId: string, fieldName: string, value: string) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  block: {
                    ...(node.data.block as BlockItem),
                    fields: (node.data.block as BlockItem).fields.map((field) =>
                      field.name === fieldName ? { ...field, value } : field
                    ),
                  },
                },
              }
            : node
        )
      );
    },
    [setNodes]
  );

  return (
    <S.Canvas ref={dropRef}>
      <ReactFlow
        nodes={nodes.map((node) => ({
          ...node,
          data: {
            ...node.data,
            onFieldChange: (fieldName: string, value: string) =>
              handleFieldChange(node.id, fieldName, value),
          },
        }))}
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
