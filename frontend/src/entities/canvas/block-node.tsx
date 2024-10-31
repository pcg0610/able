import { Handle, Position } from '@xyflow/react';
import { css } from '@emotion/react';
import { useMemo } from 'react';

// 블록 유형과 색상 매핑
const blockColors: Record<string, string> = {
  transform: '#FF6347',
  layer: '#66CDAA',
  activation: '#4682B4',
  loss: '#FFD700',
  operation: '#FF8C00',
  optimizer: '#6A5ACD',
  model: '#8A2BE2',
};

// 필드 타입 정의
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

const BlockNode: React.FC<BlockNodeProps> = ({ data }) => {
  const blockColor = useMemo(
    () => blockColors[data.type] || '#FFFFFF',
    [data.type]
  );

  const containerStyle = css`
    background-color: ${blockColor};
    padding: 10px;
    border-radius: 8px;
    width: 150px;
  `;

  return (
    <div css={containerStyle}>
      <Handle type='target' position={Position.Top} />
      <div style={{ fontWeight: 'bold', textAlign: 'center' }}>{data.type}</div>
      <div style={{ marginTop: 10 }}>
        {data.fields.map((field) => (
          <div key={field.name} style={{ marginBottom: '8px' }}>
            <label>{field.name}:</label>
            <input
              type='text'
              required={field.required}
              onChange={(e) => data.onFieldChange(field.name, e.target.value)}
              style={{ width: '100%', marginTop: '4px' }}
            />
          </div>
        ))}
      </div>
      <Handle type='source' position={Position.Bottom} />
    </div>
  );
};

export default BlockNode;
