import React, { useState, useEffect, useCallback, KeyboardEvent } from 'react';

import * as S from '@features/train/ui/sidebar/epoch-list-sidebar.style';
import { useEpochs } from '@features/train/api/use-analyze.query';
import { useProjectNameStore } from '@entities/project/model/project.model';
import { useSearchBlock } from '@features/canvas/api/use-blocks.query';

import SearchBar from '@shared/ui/input/search-bar';

const EpochListSidebar = () => {
  const [index, setIndex] = useState(0);
  const [size] = useState(10);

  const bestEpochs = ['best_train_loss', 'best_valid_loss', 'final'];

  const { projectName, resultName, epochName, setEpochName } = useProjectNameStore();
  const { data: epochData, isLoading } = useEpochs(projectName, resultName, index, size);

  const loadMoreEpochs = useCallback(() => {
    if (epochData?.hasNext && !isLoading) {
      setIndex((prevIndex) => prevIndex + 1);
    }
  }, [epochData?.hasNext, isLoading, size]);

  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const { scrollTop, clientHeight, scrollHeight } = e.currentTarget;
      if (scrollHeight - scrollTop === clientHeight) {
        loadMoreEpochs();
      }
    },
    [loadMoreEpochs]
  );

  const handleClick = (epoch: string) => {
    setEpochName(epoch);
  };

  const [value, setValue] = useState<string>('');
  const [keyword, setKeyword] = useState<string>('');

  const { data } = useSearchBlock(keyword);
  const searchBlock = data?.data.block;
  console.log(searchBlock);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleBlockSearch();
    }
  };

  const handleBlockSearch = () => {
    setKeyword(value);
  };

  const handleInputChange = (newValue: string) => {
    setValue(newValue);
    if (!newValue) {
      setKeyword('');
    }
  };

  useEffect(() => {
    if (epochData?.checkpoints && epochData.checkpoints.length > 0 && !epochName) {
      setEpochName(epochData.checkpoints[0]);
    }
  }, [epochData?.checkpoints, epochName, setEpochName]);

  return (
    <S.SidebarContainer>
      <S.BestSection>
        {bestEpochs.map((epoch, index) => (
          <S.EpochItem key={`best-${index}`} isSelected={epochName === epoch} onClick={() => handleClick(epoch)}>
            {epoch}
          </S.EpochItem>
        ))}
      </S.BestSection>
      <S.Divider />
      <SearchBar
        value={value}
        placeholder="블록 검색"
        onChange={handleInputChange}
        onClick={handleBlockSearch}
        onEnter={handleKeyDown}
      />
      <S.ScrollableSection onScroll={handleScroll}>
        {epochData?.checkpoints ? (
          epochData?.checkpoints.map((epoch, index) => (
            <S.EpochItem key={index} isSelected={epochName === epoch} onClick={() => handleClick(epoch)}>
              {epoch}
            </S.EpochItem>
          ))
        ) : (
          <div>에포크 없음</div>
        )}
        {isLoading && <div>로딩 중...</div>}
      </S.ScrollableSection>
    </S.SidebarContainer>
  );
};

export default EpochListSidebar;
