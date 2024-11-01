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

import Common from '@shared/styles/common';

const LossGraph = ({ data }) => {

   return (
      <ResponsiveContainer width="100%" height="100%">
         <ComposedChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={Common.colors.gray200} />
            <XAxis dataKey="epoch" tick={{ fontSize: Common.fontSizes['2xs'] }} />
            <YAxis domain={[0, 0.3]} tick={{ fontSize: Common.fontSizes.xs }} />
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
                  { value: 'Smoothed training loss', type: 'circle', color: Common.colors.primary },
                  { value: 'Smoothed validation loss', type: 'line', color: Common.colors.primary },
               ]}
            />
            <Area
               type="monotone"
               dataKey="validation"
               stroke={Common.colors.primary}
               fill={Common.colors.secondary}
               strokeWidth={2}
               dot={false}
               name="Smoothed validation loss"
            />
            <Line
               type="monotone"
               dataKey="training"
               stroke={Common.colors.primary}
               strokeWidth={5}
               strokeDasharray="8 4"
               dot={false}
               name="Smoothed training loss"
            />
         </ComposedChart>
      </ResponsiveContainer>
   );
};

export default LossGraph;