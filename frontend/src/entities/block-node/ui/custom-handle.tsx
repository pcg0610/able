import { Handle, HandleProps, useHandleConnections } from '@xyflow/react';

interface CustomHandleProps extends HandleProps {
  connectionCount: number;
}

const CustomHandle = (props: CustomHandleProps) => {
  const { type, connectionCount, ...rest } = props;
  const connections = useHandleConnections({ type });

  return <Handle {...rest} type={type} isConnectable={connections.length < connectionCount} />;
};

export default CustomHandle;
