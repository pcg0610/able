// result.tsx
import React from 'react';

import * as S from '@features/train/train-feature.style';

import ConfusionMatrix from '@features/train/result/confusion-matrix';
import EpochGraph from '@features/train/result/epoch-graph';
import F1Score from '@features/train/result/f1-score';
import LossGraph from '@features/train/result/loss-graph';
import PerformanceTable from '@features/train/result/performance-table';

const Result: React.FC = () => {
   return (
      <S.Container>
         <S.Header>
            <h1>모델 배포하기</h1>
            <button>모델 배포하기</button>
         </S.Header>
         <S.GridContainer>
            <S.GraphCard>
               <LossGraph />
               <S.GraphTitle>Training and validation loss</S.GraphTitle>
            </S.GraphCard>
            <S.GraphCard>
               <EpochGraph />
               <S.GraphTitle>Epoch Accuracy</S.GraphTitle>
            </S.GraphCard>
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
         </S.GridContainer>
      </S.Container>
   );
};

export default Result;
