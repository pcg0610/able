// result.tsx
import React from 'react';

import * as S from '@features/train/train-feature.style';

import ConfusionMatrix from '@features/train/result/confusion-matrix';
import EpochGraph from '@features/train/result/epoch-graph';
import F1Score from '@features/train/result/f1-score';
import LossGraph from '@features/train/result/loss-graph';
import PerformanceTable from '@features/train/result/performance-table';
import BasicButton from '@shared/ui/button/basic-button'

const Result: React.FC = () => {
   return (
      <S.Container>
         <S.Header>
            <BasicButton
               color="#0051FF"
               backgroundColor="#D8E6FB"
               text="모델 배포하기"
               onClick={() => {
                  console.log('모델 실행 버튼 클릭됨');
               }}
            />
         </S.Header>
         <S.GridContainer>
            <div className="top-row">
               <S.GraphCard>
                  <LossGraph />
                  <S.GraphTitle>Training and validation loss</S.GraphTitle>
               </S.GraphCard>
               <S.GraphCard>
                  <EpochGraph />
                  <S.GraphTitle>Epoch Accuracy</S.GraphTitle>
               </S.GraphCard>
            </div>
            <div className="bottom-row">
               <S.GraphCard>
                  <ConfusionMatrix />
                  <S.GraphTitle>Confusion Matrix</S.GraphTitle>
               </S.GraphCard>
               <S.GraphCard>
                  <F1Score />
                  <S.GraphTitle>F1-Score</S.GraphTitle>
               </S.GraphCard>
               <S.GraphCard>
                  <PerformanceTable />
                  <S.GraphTitle>Performance Matrices Table</S.GraphTitle>
               </S.GraphCard>
            </div>
         </S.GridContainer>
      </S.Container>
   );
};

export default Result;