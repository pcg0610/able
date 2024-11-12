import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

import Common from '@shared/styles/common';

interface EpochData {
  epoch: number;
  accuracy: number;
}
interface EpochGraphProps {
  epochData: EpochData[];
}

const EpochGraph = ({ epochData }: EpochGraphProps) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={epochData} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={Common.colors.gray200} />
        <XAxis dataKey="epoch" tick={{ fontSize: Common.fontSizes['2xs'] }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: Common.fontSizes.xs }} />
        <Tooltip />
        <Legend
          verticalAlign="top"
          align="right"
          height={40}
          iconSize={10}
          formatter={(value) => (
            <span style={{ color: Common.colors.graphDetail, fontSize: Common.fontSizes.xs }}>{value}</span>
          )}
          payload={[{ value: 'accuracy', type: 'line', color: Common.colors.accuracyGraph }]}
        />
        <Line
          type="monotone"
          dataKey="accuracy"
          stroke={Common.colors.accuracyGraph}
          strokeWidth={2}
          dot={false}
          name="accuracy"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default EpochGraph;
