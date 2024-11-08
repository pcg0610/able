import { ReactNode } from 'react';

import * as S from '@shared/ui/modal/modal.style';

import ModalPortal from '@shared/ui/modal/modal-portal';

interface ModalProps {
  onClose: () => void;
  isClosing?: boolean;
  onAnimationEnd?: () => void;
  title: string;
  children: ReactNode;
  cancelText?: string;
  confirmText?: string;
}

const Modal = ({
  onClose,
  isClosing,
  onAnimationEnd,
  title,
  children,
  cancelText = '취소',
  confirmText = '확인',
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
          <S.CancelButton onClick={onClose}>{cancelText}</S.CancelButton>
          <S.ConfirmButton>{confirmText}</S.ConfirmButton>
        </S.ModalFooter>
      </S.ModalWrapper>
    </S.ModalOverlay>
  </ModalPortal>
);

export default Modal;
