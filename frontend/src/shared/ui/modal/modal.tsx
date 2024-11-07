import { ReactNode } from 'react';
import * as S from '@shared/ui/modal/modal.style';
import ModalPortal from './modal-portal';

interface ModalProps {
  onClose: () => void;
  isClosing?: boolean;
  onAnimationEnd: () => void;
  title: string;
  children: ReactNode;
  CancelText?: string;
  ConfirmText?: string;
}

const Modal = ({
  onClose,
  isClosing,
  onAnimationEnd,
  title,
  children,
  CancelText = '취소',
  ConfirmText = '확인',
}: ModalProps) => (
  <ModalPortal>
    <S.ModalOverlay onClick={onClose} onAnimationEnd={onAnimationEnd} className={isClosing ? 'fadeOut' : 'fadeIn'}>
      <S.ModalWrapper onClick={(e) => e.stopPropagation()}>
        <S.ModalHeader>
          <S.Title>{title}</S.Title>
          <S.CloseButton onClick={onClose}>&times;</S.CloseButton>
        </S.ModalHeader>
        <S.ModalBody>{children}</S.ModalBody>
        <S.ModalFooter>
          <S.CancelButton onClick={onClose}>{CancelText}</S.CancelButton>
          <S.ConfirmButton>{ConfirmText}</S.ConfirmButton>
        </S.ModalFooter>
      </S.ModalWrapper>
    </S.ModalOverlay>
  </ModalPortal>
);

export default Modal;
