import { React, useState, useEffect, useCallback } from 'react';

import * as S from "@features/train/ui/sidebar/epoch-list-sidebar.style";
import { useEpochs } from '@features/train/api/use-analyze.query';
import { useProjectStateStore } from '@entities/project/model/project.model';

import SearchBox from '@shared/ui/searchbar/searchbar';

const EpochListSidebar = () => {
   const [index, setIndex] = useState(0);
   const [size] = useState(10);

   const bestEpochs = [
      "best_train_loss",
      "best_valid_loss",
      "final",
   ];

   const { projectName, epochName, setEpochName } = useProjectStateStore();
   const { data: epochData, isLoading } = useEpochs(projectName, '20241108_005251', index, size);

   const loadMoreEpochs = useCallback(() => {
      if (epochData?.hasNext && !isLoading) {
         setIndex((prevIndex) => prevIndex + 1);
      }
   }, [epochData?.hasNext, isLoading, size]);

   const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
      const { scrollTop, clientHeight, scrollHeight } = e.currentTarget;
      if (scrollHeight - scrollTop === clientHeight) {
         loadMoreEpochs();
      }
   }, [loadMoreEpochs]);


   const handleClick = (epoch: string) => {
      setEpochName(epoch);
   };

   const handleSearchChange = (value: string) => {
      console.log("Search value changed:", value);
   };

   useEffect(() => {
      if (epochData?.epochs && epochData.epochs.length > 0 && !epochName) {
         setEpochName(epochData.epochs[0]);
      }
   }, [epochData?.epochs]);

   return (
      <S.SidebarContainer>
         <S.BestSection>
            {bestEpochs.map((epoch, index) => (
               <S.EpochItem
                  key={`best-${index}`}
                  isSelected={epochName === epoch}
                  onClick={() => handleClick(epoch)}
               >
                  {epoch}
               </S.EpochItem>
            ))}
         </S.BestSection>
         <S.Divider />
         <SearchBox
            placeholder="주문번호, 상품명, 구매자명, 전화번호"
            onSearchChange={handleSearchChange}
         />
         <S.ScrollableSection onScroll={handleScroll}>
            {epochData?.epochs ? (
               epochData?.epochs.map((epoch, index) => (
                  <S.EpochItem
                     key={index}
                     isSelected={epochName === epoch}
                     onClick={() => handleClick(epoch)}
                  >
                     {epoch}
                  </S.EpochItem>
               ))
            ) : (
               <div>에포크 없음</div>
            )}
            {isLoading && <div>로딩 중...</div>}
         </S.ScrollableSection>
      </S.SidebarContainer >
   );
};

export default EpochListSidebar;
