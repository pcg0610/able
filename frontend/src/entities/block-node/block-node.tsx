import { Handle, Position } from '@xyflow/react';
import { useMemo } from 'react';

import * as S from '@entities/block-node/block-node.style';
import { capitalizeFirstLetter } from '@/shared/utils/formatters.util';

interface BlockField {
  name: string;
  required: boolean;
}

interface BlockNodeProps {
  data: {
    type: string;
    fields: BlockField[];
    onFieldChange: (fieldName: string, value: string) => void;
  };
}

const BlockNode = ({ data }: BlockNodeProps) => {
  const blockColor = useMemo(
    () => S.blockColors[data.type] || '#FFFFFF',
    [data.type]
  );
  console.log(data.fields[0].required);
  return (
    <S.Container blockColor={blockColor}>
      <Handle type='target' position={Position.Top} />
      <S.Label>{capitalizeFirstLetter(data.type)}</S.Label>
      <S.FieldWrapper>
        {data.fields.map((field) => (
          <S.InputWrapper key={field.name} blockColor={blockColor}>
            <S.Name>
              {field.name}
              {field.required}
            </S.Name>
            <S.Input
              type='text'
              placeholder={field.required ? 'required' : ''}
              required={field.required}
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
