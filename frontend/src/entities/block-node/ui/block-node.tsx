import { Position, useReactFlow } from '@xyflow/react';
import { memo, useMemo } from 'react';

import * as S from '@entities/block-node/ui/block-node.style';
import Common from '@shared/styles/common';
import { CONNECTION_LIMIT_COOUNT } from '@entities/block-node/constants/node.constant';
import { BLOCK_COLORS } from '@shared/constants/block';
import type { BlockItem } from '@features/canvas/types/block.type';
import { capitalizeFirstLetter } from '@shared/utils/formatters.util';

import CustomHandle from '@entities/block-node/ui/custom-handle';

interface BlockNodeProps {
  id: string;
  data: {
    block: BlockItem;
    isConnected?: boolean;
    isSelected?: boolean;
  };
  sourcePosition?: Position;
  targetPosition?: Position;
}

const BlockNode = ({
  id,
  data: { block, isConnected = false, isSelected = false },
  sourcePosition = Position.Bottom,
  targetPosition = Position.Top,
}: BlockNodeProps) => {
  const { updateNodeData } = useReactFlow();

  const blockColor = useMemo(() => (block?.type ? BLOCK_COLORS[block.type] : Common.colors.gray200), [block.type]);

  return (
    <S.Container blockColor={blockColor} isConnected={isConnected} isSelected={isSelected}>
      <CustomHandle type="target" position={targetPosition} connectionCount={CONNECTION_LIMIT_COOUNT} />
      <S.Label>{capitalizeFirstLetter(block?.name || 'Unknown')}</S.Label>
      <S.FieldWrapper>
        {block?.fields?.map((field) => (
          <S.InputWrapper key={field.name} blockColor={blockColor}>
            <S.Name>
              {field.name} {field.isRequired ? '*' : ''}
            </S.Name>
            <S.Input
              type="text"
              placeholder={field.isRequired ? 'required' : ''}
              required={field.isRequired}
              value={field.value || ''}
              onChange={(e) =>
                updateNodeData(id, {
                  block: {
                    ...block,
                    fields: block.fields.map((f) => (f.name === field.name ? { ...f, value: e.target.value } : f)),
                  },
                })
              }
            />
          </S.InputWrapper>
        ))}
      </S.FieldWrapper>
      <CustomHandle type="source" position={sourcePosition} connectionCount={CONNECTION_LIMIT_COOUNT} />
    </S.Container>
  );
};

export default memo(BlockNode);
