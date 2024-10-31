import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

import Common from '@shared/styles/common';

const data = [
   { name: 'Accuracy', value: 30 },
   { name: 'Top-5\nAccuracy', value: 70 },
   { name: 'Precision', value: 50 },
   { name: 'Recall', value: 90 },
];

const PerformanceTable = () => {
   return (
      <ResponsiveContainer width="100%" height="100%">
         <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 25, right: 30, left: 20, bottom: 0 }}
         >
            <CartesianGrid strokeDasharray="3 3" vertical={false} horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: Common.fontSizes.xs }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: Common.fontSizes.xs }} />
            <Tooltip />
            <Bar dataKey="value" fill="#0051FF" barSize={30} />
         </BarChart>
      </ResponsiveContainer>
   );
};

export default PerformanceTable;