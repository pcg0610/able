import {
   LineChart,
   Line,
   XAxis,
   YAxis,
   CartesianGrid,
   Tooltip,
   Legend,
   ResponsiveContainer,
} from 'recharts';

import Common from '@shared/styles/common';

const EpochGraph = ({ data }) => {
   return (
      <ResponsiveContainer width="100%" height="100%">
         <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={Common.colors.gray200} />
            <XAxis dataKey="epoch" tick={{ fontSize: Common.fontSizes.xs }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: Common.fontSizes.xs }} />
            <Tooltip />
            <Legend
               verticalAlign="top"
               align="right"
               //layout='vertical'
               height={40}
               iconSize={10}
               formatter={(value) => (
                  <span style={{ color: Common.colors.graphDetail, fontSize: Common.fontSizes.xs }}>{value}</span>
               )}
               payload={[
                  { value: 'accuracy', type: 'line', color: Common.colors.accuracyGraph },
               ]}
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