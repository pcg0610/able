import { useState } from 'react';

import * as S from '@widgets/sidebar/result-sidebar.style';

import ResultIcon from '@icons/result.svg?react'
import AnalyzeIcon from '@icons/analyze.svg?react'

interface ResultSidebarProps {
   onSelectionChange: (selection: string) => void;
}

const ResultSidebar = ({ onSelectionChange }: ResultSidebarProps) => {
   const [activeButton, setActiveButton] = useState('result');

   const handleButtonClick = (selection: string) => {
      setActiveButton(selection);
      onSelectionChange(selection);
   };

   return (
      <S.SidebarContainer>
         <S.SidebarButton
            active={activeButton === 'result'}
            onClick={() => handleButtonClick('result')}
         >
            <ResultIcon width={30} height={30} />
         </S.SidebarButton>
         <S.SidebarButton
            active={activeButton === 'analyze'}
            onClick={() => handleButtonClick('analyze')}
         >
            <AnalyzeIcon width={25} height={25} />
         </S.SidebarButton>
      </S.SidebarContainer>
   );
};

export default ResultSidebar;