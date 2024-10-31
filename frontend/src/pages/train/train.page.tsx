import { useState } from 'react';

import { Container, Content } from '@pages/train/train.style';

import Sidebar from '@widgets/sidebar/result-sidebar';
import AnalyzeComponent from '@features/train/analyze/analyze';
import ResultComponent from '@features/train/result/result';

const TrainPage = () => {
   const [selectedComponent, setSelectedComponent] = useState('result');

   const handleSidebarSelection = (selection: string) => {
      setSelectedComponent(selection);
   };

   return (
      <Container>
         <Sidebar onSelectionChange={handleSidebarSelection} />
         <Content>
            {selectedComponent === 'analyze' && <AnalyzeComponent />}
            {selectedComponent === 'result' && <ResultComponent />}
         </Content>
      </Container>
   );
};

export default TrainPage;