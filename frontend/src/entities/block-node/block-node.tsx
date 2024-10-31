// src/entities/canvas/block-node.tsx
import { Handle, Position } from '@xyflow/react';
import { useMemo } from 'react';

import {
  blockColors,
  containerStyle,
  labelStyle,
  fieldStyle,
  inputWrapperStyle,
  inputStyle,
} from './block-node.style';

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
    () => blockColors[data.type] || '#FFFFFF',
    [data.type]
  );

  return (
    <div css={containerStyle(blockColor)}>
      <Handle type='target' position={Position.Top} />
      <div css={labelStyle}>{data.type}</div>
      <div css={fieldStyle}>
        {data.fields.map((field) => (
          <div key={field.name} css={inputWrapperStyle}>
            <label>{field.name}:</label>
            <input
              type='text'
              required={field.required}
              onChange={(e) => data.onFieldChange(field.name, e.target.value)}
              css={inputStyle}
            />
          </div>
        ))}
      </div>
      <Handle type='source' position={Position.Bottom} />
    </div>
  );
};

export default BlockNode;
