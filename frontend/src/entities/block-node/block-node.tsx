import { Handle, Position } from '@xyflow/react';
import { useMemo } from 'react';

import * as S from '@entities/block-node/block-node.style';
import Common from '@shared/styles/common';
import { blockColors } from '@shared/constants/block';
import { capitalizeFirstLetter } from '@/shared/utils/formatters.util';
import { BlockItem } from '@/features/canvas/types/block.type';

interface BlockNodeProps {
  data: {
    block: BlockItem;
    onFieldChange: (fieldName: string, value: string) => void;
  };
}

const BlockNode = ({ data }: BlockNodeProps) => {
  const blockColor = useMemo(
    () =>
      data?.block?.type ? blockColors[data.block.type] : Common.colors.gray200,
    [data?.block?.type]
  );

  return (
    <S.Container blockColor={blockColor}>
      <Handle type='target' position={Position.Top} />
      <S.Label>{capitalizeFirstLetter(data?.block?.name || 'Unknown')}</S.Label>
      <S.FieldWrapper>
        {data?.block?.fields?.map((field) => (
          <S.InputWrapper key={field.name} blockColor={blockColor}>
            <S.Name>{field.name}</S.Name>
            <S.Input
              type='text'
              placeholder={field.isRequired ? 'required' : ''}
              required={field.isRequired}
              onChange={(e) => data.onFieldChange(field.name, e.target.value)}
            />
          </S.InputWrapper>
        ))}
      </S.FieldWrapper>
      <Handle type='source' position={Position.Bottom} />
    </S.Container>
  );
};

export default BlockNode;
