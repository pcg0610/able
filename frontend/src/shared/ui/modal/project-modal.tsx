import { useState } from 'react';

import * as S from '@shared/ui/modal/project-modal.style';

import DropDown from '@shared/ui/dropdown/dropdown';

interface ProjectModalProps {
   onClose: () => void;
   isClosing: boolean;
   onAnimationEnd: () => void;
   type: 'create' | 'modify';
}

const ProjectModal = ({ onClose, isClosing, onAnimationEnd, type }: ProjectModalProps) => {
   const isReadOnly = type === 'modify';
   const [selectedOption, setSelectedOption] = useState<string | null>(null);

   const options = [
      { value: 'option1', label: 'Option 1' },
      { value: 'option2', label: 'Option 2' },
      { value: 'option3', label: 'Option 3' },
      { value: 'option4', label: 'Option 4' },
      { value: 'option5', label: 'Option 5' },
   ];

   const handleSelect = (option: Option) => {
      setSelectedOption(option.label); // 선택된 옵션을 저장
      console.log('선택된 옵션:', option);
   };

   return (
      <S.ModalOverlay
         onClick={onClose}
         onAnimationEnd={onAnimationEnd}
         className={isClosing ? 'fadeOut' : 'fadeIn'}
      >
         <S.ModalWrapper onClick={(e) => e.stopPropagation()}>
            <S.ModalHeader>
               <S.Title>프로젝트 정보를 입력하세요</S.Title>
               <S.CloseButton onClick={onClose}>&times;</S.CloseButton>
            </S.ModalHeader>
            <S.ModalBody>
               <S.InputWrapper>
                  <S.Label>프로젝트 이름</S.Label>
                  <S.Input
                     placeholder={isReadOnly ? '수정 불가' : '2-50자 이내로 입력해주세요.'}
                     readOnly={isReadOnly}
                     className={isReadOnly ? 'readonly' : ''}
                  />
               </S.InputWrapper>
               <S.InputWrapper>
                  <S.Label>프로젝트 설명 (선택)</S.Label>
                  <S.Input
                     placeholder={isReadOnly ? '수정 불가' : '50자 이내로 입력해주세요.'}
                     readOnly={isReadOnly}
                     className={isReadOnly ? 'readonly' : ''}
                  />
               </S.InputWrapper>
               <S.InputWrapper>
                  <S.Label>파이썬 커널 경로</S.Label>
                  <S.Input placeholder=".exe" />
               </S.InputWrapper>
               <S.InputWrapper>
                  <S.Label>쿠다 버전</S.Label>
                  <DropDown options={options} onSelect={handleSelect} />
               </S.InputWrapper>
            </S.ModalBody>
            <S.ModalFooter>
               <S.CancelButton onClick={onClose}>취소</S.CancelButton>
               <S.ConfirmButton>{isReadOnly ? '수정' : '확인'}</S.ConfirmButton>
            </S.ModalFooter>
         </S.ModalWrapper>
      </S.ModalOverlay>
   );
};

export default ProjectModal;
