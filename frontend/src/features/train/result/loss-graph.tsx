import {
   ComposedChart,
   Line,
   XAxis,
   YAxis,
   CartesianGrid,
   Tooltip,
   Legend,
   ResponsiveContainer,
   Area,
} from 'recharts';

import { Common } from '@shared/styles/common';

const data = [
   { epoch: 0, trainingLoss: 0.02, validationLoss: 0.12 },
   { epoch: 10, trainingLoss: 0.07, validationLoss: 0.11 },
   { epoch: 20, trainingLoss: 0.05, validationLoss: 0.14 },
   { epoch: 30, trainingLoss: 0.13, validationLoss: 0.15 },
   { epoch: 40, trainingLoss: 0.10, validationLoss: 0.16 },
   { epoch: 50, trainingLoss: 0.18, validationLoss: 0.14 },
   { epoch: 60, trainingLoss: 0.20, validationLoss: 0.18 },
   { epoch: 70, trainingLoss: 0.26, validationLoss: 0.19 },
   { epoch: 80, trainingLoss: 0.28, validationLoss: 0.17 },
];

const LossGraph = () => {
   return (
      <ResponsiveContainer width="100%" height="100%">
         <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={Common.colors.gray200} />
            <XAxis dataKey="epoch" />
            <YAxis domain={[0, 0.3]} />
            <Tooltip />
            <Legend
               verticalAlign="top"
               align="right"
               formatter={(value) => (
                  <span style={{ color: Common.colors.graphDetail }}>{value}</span>
               )}
               payload={[
                  { value: 'Smoothed training loss', type: 'circle', color: Common.colors.primary },
                  { value: 'Smoothed validation loss', type: 'line', color: Common.colors.primary },
               ]}
            />
            <Area
               type="monotone"
               dataKey="validationLoss"
               stroke={Common.colors.primary}
               fill={Common.colors.secondaryButton}
               strokeWidth={2}
               dot={false}
               name="Smoothed validation loss"
            />
            <Line
               type="monotone"
               dataKey="trainingLoss"
               stroke={Common.colors.primary}
               strokeWidth={6}
               strokeDasharray="8 4"
               dot={false}
               name="Smoothed training loss"
            />
         </ComposedChart>
      </ResponsiveContainer>
   );
};

export default LossGraph;