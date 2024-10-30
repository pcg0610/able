import React from 'react';
import * as S from '@features/train/result/result.style';

const EpochGraph: React.FC = () => {
   return (
      <S.GraphContainer>
         <S.GraphTitle>Training and validation loss</S.GraphTitle>
         <S.LegendContainer>
            <S.LegendItem>
               <S.BlueDot /> Smoothed training loss
            </S.LegendItem>
            <S.LegendItem>
               <S.BlueLine /> Smoothed validation loss
            </S.LegendItem>
         </S.LegendContainer>
         {/* 임시 그래프 공간 */}
         <S.GraphPlaceholder>그래프가 여기에 표시됩니다.</S.GraphPlaceholder>
      </S.GraphContainer>
   );
};

export default EpochGraph;