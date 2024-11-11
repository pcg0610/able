import { useState, useEffect } from 'react';

import * as S from '@shared/ui/dropdown/dropdown.style';
import type { Option } from '@shared/types/common.type';

import ArrowButton from '@shared/ui/button/arrow-button';

interface DropdownProps {
  label?: string;
  options: Option[];
  placeholder?: string;
  defaultValue?: string | null;
  onSelect: (option: string) => void;
}

const Dropdown = ({ label, options, placeholder = '버전을 선택하세요', defaultValue, onSelect }: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleSelect = (option: string) => {
    setSelectedOption(option);
    setIsOpen(false);
    onSelect(option);
  };

  useEffect(() => {
    if (defaultValue && !selectedOption) {
      setSelectedOption(defaultValue);
    }
  }, [defaultValue, selectedOption]);

  return (
    <S.Container>
      {label && <S.Label>{label}</S.Label>}
      <S.DropdownWrapper>
        <S.DropdownHeader onClick={() => setIsOpen(!isOpen)} isPlaceholder={!selectedOption}>
          {selectedOption ? selectedOption : placeholder}
          <ArrowButton direction={isOpen ? 'up' : 'down'} />
        </S.DropdownHeader>
        {isOpen && (
          <S.DropdownList>
            {options.map((option) => (
              <S.DropdownItem key={option.value} onClick={() => handleSelect(option.value.toString())}>
                {option.label}
              </S.DropdownItem>
            ))}
          </S.DropdownList>
        )}
      </S.DropdownWrapper>
    </S.Container>
  );
};

export default Dropdown;
