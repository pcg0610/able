import { useState, useEffect } from 'react';

import * as S from '@shared/ui/dropdown/dropdown.style'

interface Option {
   value: string;
   label: string;
}

interface DropdownProps {
   options: Option[];
   placeholder?: string;
   defaultValue?: Option;
   onSelect: (option: Option) => void;
}

const Dropdown = ({ options, placeholder = '버전을 선택하세요', defaultValue, onSelect }: DropdownProps) => {
   const [isOpen, setIsOpen] = useState(false);
   const [selectedOption, setSelectedOption] = useState<Option | null>(defaultValue || null);

   const handleSelect = (option: Option) => {
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
      <S.DropdownContainer>
         <S.DropdownHeader
            onClick={() => setIsOpen(!isOpen)}
            isPlaceholder={!selectedOption}
         >
            {selectedOption ? selectedOption.label : placeholder}
            <S.Arrow isOpen={isOpen} />
         </S.DropdownHeader>
         {isOpen && (
            <S.DropdownList>
               {options.map((option) => (
                  <S.DropdownItem key={option.value} onClick={() => handleSelect(option)}>
                     {option.label}
                  </S.DropdownItem>
               ))}
            </S.DropdownList>
         )}
      </S.DropdownContainer>
   );
};

export default Dropdown;