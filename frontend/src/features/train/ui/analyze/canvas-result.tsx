import { useEffect, useState, useCallback } from 'react';
import {
   ReactFlow,
   MarkerType,
   useReactFlow,
   useNodesState,
   useEdgesState,
   ReactFlowProvider,
   Background,
   BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import BlockNodeFeature from '@entities/block-node/block-node-feature';
import useAutoLayout, { type LayoutOptions } from '@features/train/model/use-auto-layout.model';
import { useModel } from '@features/train/api/use-analyze.query';
import { useProjectNameStore } from '@entities/project/model/project.model';
import { useImageStore } from '@entities/train/model/train.model';
import { useFetchFeatureMap, useCreateFeatureMap } from '@features/train/api/use-analyze.mutation';
import {
   initialNodes,
   initialEdges,
} from '@/features/canvas/model/initial-data';

import { PositionedButton } from '@features/train/ui/analyze/canvas-result.style'
import BasicButton from '@shared/ui/button/basic-button'
import PlayIcon from '@icons/play.svg?react'
import { FeatureMapResponse } from '@features/train/types/analyze.type';

const proOptions = {
   account: 'paid-pro',
   hideAttribution: true,
};

const defaultEdgeOptions = {
   type: 'smoothstep',
   markerEnd: { type: MarkerType.ArrowClosed },
   pathOptions: { offset: 15 },
   animated: true,
};

const CanvasResult = () => {
   const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
   const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
   const [direction, setDirection] = useState<LayoutOptions['direction']>('TB');
   const [selectedBlockIds, setSelectedBlockIds] = useState<string[]>([]);
   const [featureMap, setFeatureMap] = useState<FeatureMapResponse[]>([]);

   const { projectName } = useProjectNameStore();
   const { uploadedImage } = useImageStore();
   const { data: canvas } = useModel(projectName, '20241108_005251');
   const { fitView } = useReactFlow();
   const { mutate: fetchCreateModel } = useCreateFeatureMap();
   const { mutate: fetchFeatureMap } = useFetchFeatureMap();
   const [autoFit, setAutoFit] = useState(false);

   useAutoLayout({ direction });

   const handleCreateModel = () => {
      fetchCreateModel(
         {
            projectName,
            resultName: '20241108_005251',
            epochName: 'epoch_1',
            deviceIndex: -1,
            image: uploadedImage,
         },
         {
            onSuccess: (data) => {
               console.log("Feature map created successfully:", data);
            },
         }
      );
   };

   const handleNodeClick = (blockId: string) => {
      setAutoFit(false);
      fetchFeatureMap(
         {
            projectName,
            resultName: '20241108_005251',
            epochName: 'epoch_1',
            blockIds: [blockId], // 클릭한 노드의 blockId를 전달
         },
         {
            onSuccess: (data) => {
               setFeatureMap(data); // featureMap 업데이트
            },
         }
      );
   };

   const handleLayoutChange = (newDirection) => {
      setAutoFit(true);
      setDirection(newDirection);
   };

   useEffect(() => {
      if (canvas) {
         const { blocks, edges } = canvas.canvas;

         const newNodes = blocks.map((block) => ({
            id: block.id,
            type: 'custom',
            position: { x: 0, y: 0 },
            data: { block, featureMap },
         }));

         const newEdges = edges.map((edge) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            ...defaultEdgeOptions,
         }));

         setNodes(newNodes);
         setEdges(newEdges);
      }
   }, [canvas, featureMap, setNodes, setEdges]);

   useEffect(() => {
      if (autoFit) {
         fitView();
      }
   }, [fitView, direction, nodes, autoFit]);

   return (
      <ReactFlow
         nodes={nodes}
         edges={edges}
         onNodesChange={onNodesChange}
         onEdgesChange={onEdgesChange}
         onNodeClick={(_, node) => handleNodeClick(node.id)}
         nodesDraggable={false}
         nodesConnectable={false}
         fitView
         attributionPosition="bottom-left"
         nodeTypes={{ custom: BlockNodeFeature }}
         proOptions={proOptions}
         zoomOnDoubleClick={false}
      >
         <Background variant={BackgroundVariant.Dots} />
         <PositionedButton>
            <BasicButton
               text="추론하기"
               icon={<PlayIcon width={13} height={15} />}
               width='10rem'
               onClick={handleCreateModel}
            />
         </PositionedButton>
         <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
            <button onClick={() => handleLayoutChange('TB')}>Down (TB)</button>
            <button onClick={() => handleLayoutChange('LR')}>Right (LR)</button>
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
