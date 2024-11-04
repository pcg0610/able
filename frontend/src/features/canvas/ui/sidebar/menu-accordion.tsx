import { ComponentType, useEffect, useRef, useState } from 'react';

import * as S from '@features/canvas/ui/sidebar/menu-accordion.style';
import { useBlocks } from '@features/canvas/api/use-blocks.query';
import { capitalizeFirstLetter } from '@/shared/utils/formatters.util';

import ArrowButton from '@shared/ui/button/arrow-button';
import MenuBlock from '@features/canvas/ui/sidebar/menu-block';

interface MenuAccordionProps {
  label: string;
  Icon?: ComponentType;
}

const MenuAccordion = ({ label, Icon }: MenuAccordionProps) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [contentHeight, setContentHeight] = useState(0);
  const contentRef = useRef<HTMLDivElement>(null);

  const { data, isFetching } = useBlocks(label, isOpen);
  const blocks = data?.data.blocks || [];

  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(contentRef.current.scrollHeight);
    }
  }, [isOpen, data]);

  const handleToggleOpen = () => {
    setIsOpen(!isOpen);
  };

  return (
    <S.Accordion>
      <S.Menu onClick={handleToggleOpen}>
        <S.LabelWrapper>
          {Icon && <Icon />}
          <span>{capitalizeFirstLetter(label)}</span>
        </S.LabelWrapper>
        <ArrowButton direction={isOpen ? 'up' : 'down'} />
      </S.Menu>
      <S.MenuBlockWrapper
        isOpen={isOpen}
        contentHeight={contentHeight}
        ref={contentRef}
      >
        {isFetching ? (
          <div>Loading...</div>
        ) : blocks.length > 0 ? (
          blocks.map((block) => (
            <MenuBlock
              key={block.name}
              type={label}
              label={block.name}
              Icon={Icon}
            />
          ))
        ) : (
          <div>No blocks available</div>
        )}
      </S.MenuBlockWrapper>
    </S.Accordion>
  );
};

export default MenuAccordion;
