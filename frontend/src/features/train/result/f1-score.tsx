import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

import Common from '@shared/styles/common';

const F1Score = ({ f1score }) => {
   const data = [
      { name: 'Score', value: f1score },
      { name: 'Remaining', value: 1 - f1score },
   ];

   return (
      <ResponsiveContainer width="100%" height="100%">
         <PieChart>
            <Pie
               data={data}
               innerRadius={55}
               outerRadius={100}
               startAngle={90}
               endAngle={-270}
               dataKey="value"
            >
               {data.map((entry, index) => (
                  <Cell
                     key={`cell-${index}`}
                     fill={index == 0 ? Common.colors.graphDetail : Common.colors.gray500}
                     fillOpacity={index === 0 ? 1 : 0.1} />
               ))}
            </Pie>
            <text
               x="50%"
               y="50%"
               textAnchor="middle"
               dominantBaseline="central"
               fontSize={Common.fontSizes.xl}
               fontWeight={Common.fontWeights.medium}
               fill="#000000"
            >
               {f1score}
            </text>
         </PieChart>
      </ResponsiveContainer>
   );
};

export default F1Score;