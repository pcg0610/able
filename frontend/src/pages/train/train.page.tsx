import React, { useState } from 'react';

import * as S from '@pages/train/train.style';

import Sidebar from '@widgets/sidebar/result-sidebar';
//import AnalyzeComponent from '@features/train/analyze';
import ResultComponent from '@features/train/result';

const TrainPage: React.FC = () => {
   const [selectedComponent, setSelectedComponent] = useState('analyze'); // 초기값을 'analyze'로 설정

   // Sidebar에서 선택된 항목 변경 함수
   const handleSidebarSelection = (selection: string) => {
      setSelectedComponent(selection);
   };

   return (
      <S.Container>
         <Sidebar onSelectionChange={handleSidebarSelection} />
         <S.Content>
            {/* {selectedComponent === 'analyze' && <AnalyzeComponent />}
            {selectedComponent === 'result' && <ResultComponent />} */}
            <ResultComponent />
         </S.Content>
      </S.Container>
   );
};

export default TrainPage;