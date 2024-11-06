import { useCallback, useEffect, useState } from 'react';
import {
   ReactFlow,
   MarkerType,
   useReactFlow,
   useNodesState,
   useEdgesState,
   ReactFlowProvider,
   addEdge,
   ConnectionLineType,
   Background,
   BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import BlockNode from '@entities/block-node/block-node';
import useAutoLayout, { type LayoutOptions } from '@features/train/model/use-auto-layout.model';

import { PositionedButton } from '@features/train/ui/analyze/canvas-result.style'
import BasicButton from '@shared/ui/button/basic-button'
import PlayIcon from '@icons/play.svg?react'

const initialNodes = [
   {
      id: 'horizontal-1',
      type: 'custom',
      position: { x: 0, y: 80 },
      data: {
         block: {
            type: 'data',
            name: 'Input',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-2',
      type: 'custom',
      position: { x: 250, y: 0 },
      data: {
         block: {
            type: 'activation',
            name: 'relu',
            fields: [
               { name: 'inplace', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-3',
      type: 'custom',
      position: { x: 250, y: 160 },
      data: {
         block: {
            type: 'layer',
            name: 'linear',
            fields: [
               { name: 'in_features', isRequired: true },
               { name: 'out_features', isRequired: true },
               { name: 'bias', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-4',
      type: 'custom',
      position: { x: 500, y: 0 },
      data: {
         block: {
            type: 'data',
            name: 'Node 4',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-5',
      type: 'custom',
      position: { x: 500, y: 100 },
      data: {
         block: {
            type: 'data',
            name: 'Node 5',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-6',
      type: 'custom',
      position: { x: 500, y: 230 },
      data: {
         block: {
            type: 'data',
            name: 'Node 6',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-7',
      type: 'custom',
      position: { x: 750, y: 50 },
      data: {
         block: {
            type: 'data',
            name: 'Node 7',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
   {
      id: 'horizontal-8',
      type: 'custom',
      position: { x: 750, y: 300 },
      data: {
         block: {
            type: 'data',
            name: 'Node 8',
            fields: [
               { name: 'data_path', isRequired: true },
               { name: 'input_shape', isRequired: true },
               { name: 'classes', isRequired: true },
            ],
         },
      },
   },
];

const initialEdges = [
   {
      id: 'horizontal-e1-2',
      source: 'horizontal-1',
      target: 'horizontal-2',
   },
   {
      id: 'horizontal-e1-3',
      source: 'horizontal-1',
      target: 'horizontal-3',
   },
   {
      id: 'horizontal-e1-4',
      source: 'horizontal-2',
      target: 'horizontal-4',
   },
   {
      id: 'horizontal-e3-5',
      source: 'horizontal-3',
      target: 'horizontal-5',
   },
   {
      id: 'horizontal-e3-6',
      source: 'horizontal-3',
      target: 'horizontal-6',
   },
   {
      id: 'horizontal-e5-7',
      source: 'horizontal-5',
      target: 'horizontal-7',
   },
   {
      id: 'horizontal-e6-8',
      source: 'horizontal-6',
      target: 'horizontal-8',
   },
];

const proOptions = {
   account: 'paid-pro',
   hideAttribution: true,
};

const defaultEdgeOptions = {
   type: 'smoothstep',
   markerEnd: { type: MarkerType.ArrowClosed },
   pathOptions: { offset: 5 },
   animated: true,
};

const CanvasResult = () => {
   const [nodes, _, onNodesChange] = useNodesState(initialNodes);
   const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
   const [direction, setDirection] = useState<LayoutOptions['direction']>('TB');

   const onConnect = useCallback(
      (params) => setEdges((els) => addEdge(params, els)),
      [],
   );

   const { fitView } = useReactFlow();

   useAutoLayout({ direction });

   useEffect(() => {
      console.log('Direction changed:', direction);
      fitView();
   }, [nodes, fitView, direction]);

   return (
      <ReactFlow
         nodes={nodes}
         edges={edges}
         onNodesChange={onNodesChange}
         onEdgesChange={onEdgesChange}
         onConnect={onConnect}
         nodesDraggable={false}
         fitView
         attributionPosition="bottom-left"
         defaultEdgeOptions={defaultEdgeOptions}
         nodeTypes={{ custom: BlockNode }}
         connectionLineType={ConnectionLineType.SmoothStep}
         proOptions={proOptions}
         zoomOnDoubleClick={false}
      >
         <Background variant={BackgroundVariant.Dots} />
         <PositionedButton>
            <BasicButton
               text="추론하기"
               icon={<PlayIcon width={13} height={15} />}
               onClick={() => {
                  console.log('모델 실행 버튼 클릭됨');
               }}
            />
         </PositionedButton>
         <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
            <button onClick={() => setDirection('TB')}>Down (TB)</button>
            <button onClick={() => setDirection('LR')}>Right (LR)</button>
            <button onClick={() => setDirection('BT')}>Up (BT)</button>
            <button onClick={() => setDirection('RL')}>Left (RL)</button>
         </div>
      </ReactFlow>
   );
};

const WrappedCanvasResult = () => (
   <ReactFlowProvider>
      <CanvasResult />
   </ReactFlowProvider>
);

export default WrappedCanvasResult;