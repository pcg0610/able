// ProjectModal.tsx
import * as S from '@shared/ui/modal/project-modal.style';

interface ProjectModalProps {
   onClose: () => void;
   isClosing: boolean;
   onAnimationEnd: () => void;
   type: 'create' | 'modify';
}

const ProjectModal = ({ onClose, isClosing, onAnimationEnd, type }: ProjectModalProps) => {
   const isReadOnly = type === 'modify';

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
                  <S.Select>
                     <option>버전을 선택해주세요.</option>
                     <option>버전 1</option>
                     <option>버전 2</option>
                     <option>버전 3</option>
                  </S.Select>
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
