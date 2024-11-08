import { KeyboardEvent, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/canvas-sidebar.style';
import { BLOCK_MENU } from '@features/canvas/costants/block-menu.constant';

import SearchBar from '@shared/ui/input/search-bar';
import MenuAccordion from '@features/canvas/ui/sidebar/menu-accordion';

const CanvasSidebar = () => {
  const [value, setValue] = useState<string>('');

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleBlockSearch();
    }
  };

  const handleBlockSearch = () => {
    console.log(value);
  };

  return (
    <S.SidebarContainer>
      <SearchBar
        value={value}
        placeholder="블록 검색"
        onChange={setValue}
        onClick={handleBlockSearch}
        onEnter={handleKeyDown}
      />
      {BLOCK_MENU.map((menu) => (
        <MenuAccordion key={menu.name} label={menu.name} Icon={menu.icon} />
      ))}
    </S.SidebarContainer>
  );
};

export default CanvasSidebar;
