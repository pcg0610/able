import React, { useState } from 'react';

import * as S from '@widgets/sidebar/style/result-sidebar.style';

type ResultSidebarProps = {
   onSelectionChange: (selection: string) => void;
};

const ResultSidebar: React.FC<ResultSidebarProps> = ({ onSelectionChange }) => {
   const [activeButton, setActiveButton] = useState('analyze');

   const handleButtonClick = (selection: string) => {
      setActiveButton(selection);
      onSelectionChange(selection);
   };

   return (
      <S.SidebarContainer>
         <S.SidebarButton
            active={activeButton === 'analyze'}
            onClick={() => handleButtonClick('analyze')}
         >
            📊
         </S.SidebarButton>
         <S.SidebarButton
            active={activeButton === 'result'}
            onClick={() => handleButtonClick('result')}
         >
            📈
         </S.SidebarButton>
      </S.SidebarContainer>
   );
};

export default ResultSidebar;